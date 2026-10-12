"""Original-cell narrative templates for clean image reports.

Historical code is replaced explicitly by report-only reads. Image fitting,
checkpoint inference and reserved-test evaluation are never notebook actions.
"""
from __future__ import annotations

from .dedup_notebook_templates import original_cells, template_sections


def _markdown(name, updates=None):
    result = {index: cell.source for index, cell in enumerate(original_cells(name), 1)
              if cell.cell_type == "markdown"}
    for index, replacements in (updates or {}).items():
        if isinstance(replacements, str):
            result[index] = replacements
        else:
            for old, new in replacements:
                if old not in result[index]:
                    raise ValueError(f"Stale original prose: {name}, cell {index}: {old[:70]}")
                result[index] = result[index].replace(old, new)
    links = {
        "10_image_architecture_comparison.ipynb": "10_bounded_image_pilots.ipynb",
        "11_image_augmentation.ipynb": "10_bounded_image_pilots.ipynb",
        "12_image_lighting.ipynb": "10_bounded_image_pilots.ipynb",
        "13_image_uncertainty.ipynb": "11_image_uncertainty.ipynb",
        "14_image_nested_evaluation.ipynb": "12_combined_evaluation.ipynb",
        "09_image_b0_pilot.ipynb": "09_image_modelling.ipynb",
        "../concepts.md": "../concepts.md",
    }
    for index in result:
        for old, new in links.items():
            result[index] = result[index].replace(old, new)
    return result


IMAGE_SETUP = """from melanoma_detection import dedup_notebook_images as image_docs
from melanoma_detection.image_nested_calibration_reporting import load_saved_calibration_run
image_docs.original_style('image')
raw = load_image_report(ROOT)
calibration = load_saved_calibration_run(ROOT, image_docs.CALIBRATION_RUN)
display(pd.DataFrame([{'stage': 'raw five-fold B0', 'status': raw['status'], 'complete': raw['complete']},
    {'stage': 'training-only calibration', 'status': calibration['status'], 'complete': calibration['complete']}]))"""


def _report_setup(stage, code):
    from .dedup_notebooks import SETUP
    return f"NOTEBOOK_STAGE = {stage!r}\n" + SETUP + "\n" + code


def nested_sections(*, setup=True):
    name = "14_image_nested_evaluation.ipynb"
    md = _markdown(name, {
        1: [("# 14 — Image model: five outer folds", "# Clean image model: five preserved outer folds")],
        3: [("canonical d1\ndevelopment CSV", "frozen dedup_v1\ndevelopment manifest; all 422 original patient assignments are preserved"),
            ("1. Build three inner patient folds.", "1. Preserve the three recorded inner patient folds after removing exact copies."),
            ("Each canonical\nimage", "Each retained\nimage"),
            ("Earlier architecture, augmentation and recipe pilots used this development\ndataset, and archived exploratory predictions used outer fold 0.",
             "Earlier architecture, augmentation and recipe pilots used the original development\npopulation, and archived exploratory predictions used outer fold 0. Exact-copy removal\ndoes not undo those earlier choices or make these patients newly untouched.")],
        7: [("complete canonical development cohort", "complete retained development cohort: 6,706 images, 422 patients,\n127 malignant images and 88 patients with a malignant image")],
        11: [("No new threshold or calibration model is fitted here.", "No new threshold or calibration model is fitted by this report. The ten fixed-width\nbins, image-count annotations and score histogram retain the original composition.\nThe added confusion panel reports both counts and percentages within true-class rows,\nusing the same blue scale as the original pilot notebooks.")],
        13: [("It may start after the raw\nfive-fold evaluation finishes, if three fresh fixed-duration inner refits per\nouter fold fit the same session deadline.",
              "Its clean artifacts are read only after the raw\nfive-fold evaluation verifies. Three fresh fixed-duration inner refits per\nouter fold were the separate training-only calibration protocol; opening this\nnotebook starts none of those fits."),
             ("All five calibration folds must finish before the comparisons below appear.", "All five calibration folds must finish and their clean receipts must verify\nbefore the comparisons below appear; the status table reports actual completion.")],
        16: [("existing hash-recorded OOF\nprediction bundle with verified canonical identities", "clean hash-recorded OOF\nprediction bundle with verified retained identities")],
    })
    code = {
        2: IMAGE_SETUP if setup else "image_docs.original_style('image')",
        4: "image_docs.image_settings(ROOT)",
        6: "image_docs.image_selection_table(ROOT, data)",
        8: "if raw['complete']:\n    display(raw['fold_metrics'][['fold', 'n_images', 'n_patients', 'n_positive', 'prevalence']])\nelse:\n    display(Markdown('**Pending:** full retained population counts require five verified folds.'))",
        10: "image_docs.raw_outer_ranking(raw, finish_figure)",
        12: "image_docs.raw_outer_reliability(raw, finish_figure)",
        14: "image_docs.calibration_figures(calibration, finish_figure)",
        15: "if calibration['complete']:\n    display(calibration['ranking_invariance'])\n    display(calibration['scenario_by_fold'])",
        17: "image_docs.matched_comparator_figures(score_bundle(), finish_figure)",
        19: "image_docs.image_takeaways(raw, comparison_table())",
    }
    return template_sections(name, code, md,
        context_note="**Scope change:** exact copies are removed before fitting and evaluation.\nThe five-fold design matches the original nested-evaluation report; original pilot\noutputs are never substituted for missing clean results.")


def modelling_sections():
    name = "09_image_b0_pilot.ipynb"
    md = _markdown(name, {
        1: [("# 09 — Image model: the six-epoch B0 pilot", "# 09 — Clean image model: B0 screening and five-fold evaluation"),
            ("We report each recipe's saved checkpoint\nwith the highest validation average precision (AP).", "The saved first-inner-fold histories show each recipe's six-epoch screen.\nAverage precision (AP) chooses the screening recipe; three twelve-epoch\nconfirmations then determine the fixed duration of a fresh outer-training refit."),
            ("Full out-of-fold (OOF) evaluation would test the complete\nselection process on independent outer folds; the reserved test comes last.\nOuter fold 0 and the reserved test are excluded here.", "The second part of this notebook reads full out-of-fold (OOF) evaluation: the\ncomplete selection process repeats inside each outer-training pool, then scores\nits held-out patients. Outer fold 0 is excluded from the first pool's selection\nexamples, but later supplies its independent outer prediction. The reserved test\ncomes last and stays excluded from every section."),
            ("Original outputs are preserved; new reports use `b0_pilot_reporting_v2`.", "Original outputs are preserved; raw clean results use `b0_nested_dedup_v1`,\nand report figures belong only to the numbered clean-workflow directory.")],
        2: [("This protects the meaning of the historical results after code changes.", "This protects the identity of the retained development predictions after code changes.\nThe clean manifest records 6,706 images rather than the original 6,864; all\n422 patients and their five original outer groups remain present.")],
        4: [("whose labels are\nnot used here", "whose labels are\nnot used for that outer pool's recipe and duration selection")],
        8: [("All four recipes ran six epochs", "In every outer-training pool all four recipes were screened for six epochs"),
            ("Exact AP ties retain the earlier checkpoint.", "Exact AP ties retain the earlier checkpoint within a fit. Screening recipes\nwithin 0.0001 of maximum AP tie, preferring no weighting then lower learning rate.\nThe median AP-best epoch across three confirmations fixes each fresh outer refit.")],
        10: [("A late AP peak motivates further checking, not proof that longer training helps.", "A late AP peak motivates further checking, not proof that longer training helps.\nThe plotted four-recipe screen is restricted to epochs 1–6 even when its selected\ntrajectory subsequently continued to twelve. This keeps the original learning-curve\ncomparison on a common budget; the separate selection table documents all confirmations.")],
        12: [("each line is a recipe's selected checkpoint on the same\n1,834 images from 113 patients", "the original two-panel ROC/precision–recall composition now shows the\nverified retained outer predictions on matching records"),
             ("These patients\nalso helped select the checkpoints, so the comparison remains exploratory.", "**Scope change:** inner-screen best-checkpoint predictions were not exported\nby the clean bounded runner, so these are outer OOF curves rather than the original\none-split pilot curves. No missing inner predictions are reconstructed by inference.\nPooled curves are secondary; unweighted mean-fold metrics remain the primary summary.")],
        14: [("here.\nRecall/sensitivity", "here. This report uses verified retained outer predictions, while the original\npilot used selected inner-validation checkpoints.\nRecall/sensitivity")],
        16: [("each panel is a recipe", "the panel describes the verified raw outer predictions"),
             ("grouped into equal-count bins", "grouped into the original nested report's ten fixed-width bins"),
             ("The shared axes cover the\nobserved bin means, so check their range before comparing other plots.", "The reliability panel retains the full [0,1] range, with bin image counts\nannotated; the adjacent histogram shows the score distribution on the same range.\n**Scope change:** this follows the original five-fold report's fixed bins instead\nof substituting its pilot's five equal-count recipe panels.")],
        18: """## Interpretation and next steps

The saved clean screen and confirmation table reports each selected recipe and
its three AP-best epochs. Read the executed recap below instead of carrying
forward the original pilot's AP 0.1383 or preferred epoch: those belonged to
different, duplicate-containing fitting and validation records. A late screen
peak can motivate checking duration on further patient splits, but does not
establish an optimal duration or make B0 the final architecture.

[Notebook 10](10_bounded_image_pilots.ipynb) documents the narrower clean duration
and ConvNeXt-Tiny pilots, with DenseNet, stronger augmentation and lighting
correction explicitly deferred. [Notebook 11](11_image_uncertainty.ipynb) reads
clean five-fold uncertainty diagnostics. The following nested-evaluation sections
verify B0's five outer refits and the separate training-only calibration records.
Any fusion and the independent reserved test remain later work after the entire
procedure, calibration choice and decision rule are settled.""",
        20: [("- A finite image preview cannot certify every lesion; full outer evaluation and the reserved test remain pending.", "- A finite image preview cannot certify every lesion; the verified full outer report follows below and the reserved test remains pending.\n- Exact-copy removal changes both fitting weight and evaluation weight; it does not undo earlier development exploration.")],
        21: [("The saved six-epoch experiment provides a verified B0 reference.", "The saved retained six-epoch screens provide verified B0 selection diagnostics;\nthe five-fold report below evaluates the resulting bounded procedure.")],
    })
    code = {
        3: _report_setup("09_image_modelling", IMAGE_SETUP),
        5: "image_docs.image_partition_table(ROOT, data, dataset_identity)\nimage_docs.image_settings(ROOT)",
        7: "image_docs.show_preparation(ROOT, data, dataset_identity, finish_figure)",
        9: "image_docs.image_selection_table(ROOT, data)\nif raw['complete']:\n    display(raw['fold_metrics'])",
        11: "image_docs.image_progress(ROOT, data, dataset_identity, finish_figure)",
        13: "if raw['complete']:\n    image_docs.plot_probability_views(raw['predictions'], {'Raw B0': 'base_probability'}, finish_figure, 'raw_b0', 'Clean B0 outer OOF: original pilot curve composition', reliability=False)",
        15: "if raw['complete']:\n    image_docs.plot_confusions(raw['predictions'], {'Raw score ≥ 0.5 (descriptive)': raw['predictions'].base_probability.ge(.5)}, finish_figure, 'raw_b0')",
        17: "image_docs.raw_outer_reliability(raw, finish_figure)",
        19: "image_docs.image_takeaways(raw, comparison_table())",
        22: "image_docs.image_selection_table(ROOT, data)",
    }
    return template_sections(name, code, md,
        context_note="**Scope change:** original six-epoch pilot explanations remain beside the\nclean screening evidence; full five-fold evaluation follows in the original\nnested-report section order. These two evidence scopes are labelled separately.") + nested_sections(setup=False)


def pilot_sections():
    name = "10_image_architecture_comparison.ipynb"
    md = _markdown(name, {
        1: [("# 10 — Compare image-model architectures", "# 10 — Bounded clean image pilots and deferred transformation experiments"),
            ("This notebook compares pretrained convolutional neural networks on matching\npatients:", "The original template compares pretrained convolutional neural networks on\nmatching patients. The clean workflow narrows the available comparison to B0\nduration and ConvNeXt-Tiny; DenseNet121 has no clean pilot result. For context:"),
            ("For each recipe, the highest validation AP checkpoint is reported.", "Fixed epoch-12 and epoch-24 endpoints are reported separately from the\nhighest-validation-AP checkpoint within each cap. Comparing fixed endpoints\nand selected checkpoints answers different questions."),
            ("Eight new fits were budgeted; the saved completion\nledger establishes what actually ran.", "The clean protocol proposes three B0 duration fits (72 epochs) and six\nConvNeXt fits (90 epochs, including four six-epoch screens). The saved completion\nledger establishes what actually ran; absent runs remain pending."),
            ("Training defaults to disabled.", "Training is not part of notebook execution.")],
        3: [("Each admitted fit runs six epochs, including\none classifier-only warmup, under a fixed 30-epoch learning-rate schedule.", "ConvNeXt screens four recipes for six epochs on inner fold 0, then confirms\nthe chosen recipe through 24 epochs on all three preserved inner folds. B0\nuses the clean baseline's selected outer-pool recipe for three 24-epoch fits.\nThe fixed 30-epoch learning-rate schedule and one classifier-only warmup remain;\nthe duration endpoints are 12 and 24, not the original common six-epoch cap."),
            ("B0/DenseNet keep", "B0 keeps"),
            ("Only completed\nmatching slots rank families", "Only completed\nmatching retained inner records enter the available two-family comparison"),
            ("The\noriginal session allowed at most 50 training minutes within its wider deadline.", "The original three-family, six-epoch resource cap is historical. The clean\nledger records its own executed fits, recorded minutes and memory; no budget\nor result is inferred from the older experiment.")],
        7: [("Each line is a recipe; paused runs are labelled.", "Each line is a family and preserved inner split from a selected trajectory;\nsolid lines are B0 and dashed lines are ConvNeXt-Tiny. Missing families have no line."),
            ("Six epochs\nshow early progress, not each family's full learning potential.", "Vertical 12/24 guides distinguish the prescribed endpoint caps from the\nwarmup guide. Twenty-four epochs show the bounded trajectory, not each family's\nfull learning potential. Best-in-cap checkpoints reuse these validation labels,\nso their metrics remain selected exploratory evidence.")],
        9: [("Only slots completed for **all three families** enter these tables and plots.", "**Scope change:** only completed B0/ConvNeXt-Tiny clean runs enter these tables\nand plots, and an actual architecture contrast requires both. DenseNet is deferred;\na missing family cannot lose a comparison. Fixed endpoint metrics and AP-selected\nbest-in-cap metrics are shown separately, averaged over three inner folds."),
            ("The three panels\nshow selected-checkpoint AP (longer is better), total measured training minutes\nand peak allocated GPU memory in GiB (shorter is cheaper).", "The endpoint AP panels keep fixed 12/24 and AP-selected checkpoints separate;\nthe adjacent resource panels show total measured training minutes and peak\nallocated GPU memory in GiB (shorter is cheaper)."),
            ("**Read each matched-slot curve pair:**", "**Read the matching fixed-24 curve pair:**")],
        13: [("The table below lists any completed unmatched trials separately.", "The status table records complete, pending or budget-stopped clean families separately."),
             ("Next, validate every family with longer training and three inner patient folds.", "The bounded clean pilot already prescribes three inner patient folds for each\nadmitted family. Complete that protocol before considering any wider family search."),
             ("Fusion, final outer evaluation, any eligible Swin experiment\nand reserved-test evaluation remain separate work.", "Fusion, full alternative-family outer evaluation, any eligible Swin experiment\nand reserved-test evaluation remain separate work. Clean B0's distinct five-fold\nprocedure is reported in notebook 09, without promoting a pilot candidate automatically.")],
        15: [("- Six epochs on one inner split measure early learning rather than final potential.", "- The recipe screen uses one inner split, while selected 12/24 trajectories use three; neither establishes final architecture performance."),
             ("- Longer confirmation, calibration, full outer evaluation, fusion and the reserved test remain pending.", "- Full challenger outer evaluation, stronger transformations, fusion and the reserved test remain pending.\n- Clean B0 calibration and outer evaluation have their own verified reports; they cannot confer that evidence on an untested challenger.")],
    })
    code = {
        2: _report_setup("10_bounded_image_pilots", "from melanoma_detection import dedup_notebook_images as image_docs\nimage_docs.original_style('image')\npilots = image_docs.load_pilot_bundle(pilot_report)"),
        4: "image_docs.pilot_screening(pilots)\nimage_docs.image_partition_table(ROOT, data, dataset_identity)",
        6: "image_docs.show_preparation(ROOT, data, dataset_identity, finish_figure)",
        8: "image_docs.pilot_learning(pilots, finish_figure)",
        10: "image_docs.pilot_endpoint_metrics(pilots, finish_figure)\nimage_docs.pilot_cost(pilots, finish_figure)\nimage_docs.pilot_matching_curves(pilots, finish_figure)",
        12: "image_docs.pilot_decisions(pilots, finish_figure)",
        14: "image_docs.pilot_interval_support(pilots, finish_figure)",
        17: "display(Markdown('**Qualified conclusion:** a completed selected-inner pilot is a candidate procedure, not a final architecture. Missing clean families and transformation conditions supply no performance comparison.'))",
    }
    sections = template_sections(name, code, md,
        context_note="**Genuine experiment-scope change:** the original three-family sweep is\nreplaced by a bounded B0 duration/ConvNeXt-Tiny proposal on three preserved inner\nfolds. DenseNet, stronger augmentation, CLAHE and wider preparation searches\nare deferred. All original explanatory sections remain, adapted below.")
    return sections + deferred_condition_sections("11_image_augmentation.ipynb", "stronger augmentation") + deferred_condition_sections("12_image_lighting.ipynb", "CLAHE lighting correction")


def deferred_condition_sections(name, condition):
    cells = original_cells(name)
    md = _markdown(name)
    md[1] = md[1].replace("# 11 —", "# Deferred clean experiment —").replace("# 12 —", "# Deferred clean experiment —")
    md[1] = md[1].replace("This notebook reads saved results for a controlled B0 experiment.",
        "This section preserves the original controlled-experiment rationale as a\ndeferred clean-workflow method. No matching clean control/candidate fits,\nquality approval or prediction exports exist for this transformation comparison.")
    status_pos = md[1].find("**Current saved status:**")
    if status_pos >= 0:
        md[1] = md[1][:status_pos] + """**Current clean status: deferred.** The original weak-control execution and
quality-blocked stronger candidate belong to the duplicate-containing historical
workflow. They are not clean results and are not copied into this report. A new
candidate must pass a retained-fitting-image quality review and then complete
matching fresh fits before any improvement or deterioration can be reported."""
    md[3] = md[3].replace("The saved first inner fold lies within outer-training folds 1–4: 3,660 fitting\nimages from 225 patients and 1,834 validation images from 113 patients.",
        "Any future clean comparison must use the preserved first inner fold within\nouter-training folds 1–4 after exact-copy removal. The clean partition table\nabove reports its actual fitting and validation support; the original\n3,660/1,834 image counts are historical and do not define a new clean cohort.")
    md[3] += "\n\n**Deferred protocol qualification:** the recipe described here is the original\ncontrolled-reference design, not a declaration that its settings are selected\nfor a new clean candidate. Record any approved recipe change before fitting,\nand preserve original patient memberships, transformation comparisons and\nquality criteria when making that later experiment concrete."
    md[5] = md[5].replace("The frozen quality sample contains", "The original quality-gate template prescribes").replace("Each saved page has", "Each future review page should have")
    md[5] = md[5].replace("The saved review blocked B; this is a quality decision,", "The historical review blocked B; no fresh clean review is inferred. This was a quality decision,")
    md[5] = md[5].replace("That gate was not met: there are no saved C preview\npages, and the following missing-preview message is expected.",
        "The historical gate was not met, and no clean A/B prerequisite or C review\nhas been completed. There are no clean C preview pages or fit records; the\nexplicit deferred-status table below records that missing experiment.")
    md[5] += "\n\nNo historical quality page is relabelled as a clean review. Inspect retained\nfitting examples and record explicit approval for the exact new transformation\nprofile before any candidate run. The current weak preview documents the\nunchanged clean control preparation only."
    md[7] = md[7].replace("Only committed epochs and complete twelve-epoch prediction records are shown.\nEach line names an available condition; a missing condition has no performance\nresult. The current saved figures show the weak control A only, so they cannot\nestablish an A/B or A/C winner.",
        "No clean transformation-condition curves are available, so this section\nshows an explicit deferred status instead of historical figures. Once a later\nexperiment is approved, only committed epochs and complete matching twelve-epoch\nexports may enter the following original figure sequence. A missing condition\nhas no performance result and cannot establish an A/B or A/C winner.")
    md[7] = md[7].replace("**Read the figures in order:**", "**Reading guide retained for the future matching report:**")
    md[11] += "\n\n**No clean numerical conclusion:** neither the screening margin nor a historical\ncontrol score creates an AP difference for this deferred experiment. The saved\nstatus table records no clean candidate comparison, and no training is launched\nby any cell in this section."
    md[13] = md[13].replace("Current [uncertainty diagnostics]", "Clean [uncertainty diagnostics]")
    md[14] = md[14].replace("One inner split contains only 33 malignant validation images", "The original one-split pilot contained only 33 malignant validation images;\na future clean report must use its own verified malignant support").replace("The saved control\nprovides a reference", "The original saved control\nprovides historical context only")
    md[14] += "\n\nThe broader clean B0 five-fold evaluation is already a different experiment.\nIts completion does not make this deferred transformation comparison complete,\nand its raw or calibrated scores are not substitutes for matching candidate fits."
    code = {i: f"image_docs.deferred_condition_status({condition!r})" for i, c in enumerate(cells, 1) if c.cell_type == "code"}
    return template_sections(name, code, md,
        context_note=f"**Scope change — deferred {condition}.** This method and its original\nreading guides are retained for documentation; no historical control, candidate\nmetric, quality approval or prediction file is presented as a clean result.")


def uncertainty_sections():
    name = "13_image_uncertainty.ipynb"
    md = _markdown(name, {
        1: [("# 13 —", "# 11 — Clean outer-fold"),
            ("We load an existing\n**checkpoint** (saved model weights) and perform inference only. The default is\nthe weak-augmentation B0 control, using its recorded best epoch rather than the\nfinal epoch 12.", "The saved clean inference used each image's own outer-held-out B0\n**checkpoint** (saved model weights), from a fresh fixed-duration refit. The\nreport loads exported predictions and receipts only: it does not rebuild a\nnetwork, choose an old best checkpoint or perform inference."),
            ("These are **inner-validation diagnostics**: patients excluded from fitting also\nhelped choose the checkpoint and pilot recipes. Direct training leakage is\nchecked, but that selection still makes results exploratory. Full out-of-fold\n(OOF) evaluation would test the complete selection process on independent outer\npatients; the reserved test comes last. Neither is performed here, and a pilot\nAP leader does not become the final architecture.", "These are **five-fold outer-held-out diagnostics**: each scored patient's\nimages were excluded from that checkpoint's fitting, recipe and duration selection.\nThis strengthens the original one-inner-split scope. Earlier development\nexploration remains a limitation; uncertainty rankings still do not select a\nclinical referral policy or final architecture. The reserved test comes last\nand is never opened by this report."),
            ("Every current execution writes a new directory\nunder `results/image_uncertainty_runs/`.", "Clean pass exports belong to `results/dedup_v1/uncertainty/b0_uncertainty_dedup_v1`;\nexecuting this notebook regenerates only its separate clean presentation figures.")],
        2: """## 1. Choose saved clean checkpoints explicitly

The clean inference receipts identify one fresh outer refit per preserved fold.
They record the exact run, recipe (training settings), selected fixed duration,
checkpoint hash (file fingerprint), source-image identities, deterministic
preparation and saved ordinary predictions. All five raw refits must complete
before full clean uncertainty is accepted. Missing or incompatible artifacts
stop verification rather than silently falling back to a historical model.

**Scope change:** the original selectable one-pilot checkpoint adapter is replaced
by recorded `b0_nested_dedup_v1` outer checkpoints. This is an explicit matching
clean inference scope, not automatic architecture selection. `SOURCE_RUN`,
architecture and recipe are provenance for every fold, not notebook settings
that trigger another inference run. The table below reads actual completion;
no checkpoint is loaded into a neural network by this notebook.""",
        4: """## 2. Recover the exact evaluation patients

Use the frozen retained development manifest and every preserved outer fold.
Each image receives prediction from its own outer-held-out checkpoint; no
patient occurs in that checkpoint's fitting or inner selection membership.
Patient IDs define separation and bootstrap groups; they are not model inputs.
Recorded identities and receipts bind source images, checkpoint bytes and the
ordinary prediction export, rather than recreating a newly sampled split.

The clean cohort contains 6,706 images from 422 patients, including 127 malignant
images from 88 patients with malignancy. **Scope change:** these replace the
historical weak-control pilot's 1,834 validation images from 113 patients.
Outer fold 0 is now scored by its matching outer refit rather than excluded
from the uncertainty report. Reserved-test tables and images remain outside
the loader and every diagnostic shown here.""",
        6: [("Any failure stops execution before the 30-pass run.", "Any failure stopped the separate clean inference runner before its 30-pass run.\nThis notebook reads its recorded smoke checks; it does not rerun inference or\nmutate model state. The completed receipts are shown below when available.")],
        14: [("with separate lines for mean score and SD", "with separate solid/ dashed lines for mean score and SD for each preserved fold"),
             ("The table also gives overlap among the highest-SD 20%.", "The table also gives overlap among the highest-SD 20%. Fold colours identify\nthe five different held-out checkpoints; the original two-panel composition\nand numerical-versus-ranking distinction are retained.")],
        16: [("The checkpoint\nwas chosen on this split, so these checks may be optimistic.", "Each checkpoint's duration was chosen within its own training patients.\n**Scope change:** these pooled diagrams use matching clean outer-held-out\npredictions, while unweighted mean-fold AP/AUC remain primary. The earlier\npilot/development adaptations still limit the independence of final claims.")],
        23: [("the 113 validation patients", "the original number of retained patients independently within each of the five outer folds"),
             ("do not remove optimism\nfrom checkpoint selection on this split", "do not include retraining or repeat the recipe/duration selection process,\nand do not undo earlier development adaptation")],
        26: [("before and after inference", "in the saved before/after inference receipts"),
             ("Raw, summary and figure\noutputs belong only to this new run directory.", "Raw and summary outputs belong only to the clean inference directory.\nFigure outputs belong to this notebook's separate clean report directory;\nno source checkpoint, saved raw pass or historical result is overwritten.")],
        28: [("This is one small, imbalanced inner-validation split reused for epoch and pilot\nrecipe selection. Patient separation prevents direct training leakage but does\nnot make these diagnostics an unbiased final evaluation.", "This is a complete but imbalanced retained five-fold development cohort.\nEach scored patient is excluded from its own checkpoint's recipe/duration\nselection, and direct training leakage is checked. Earlier development pilots\nand model exploration mean this still cannot replace an independent final test."),
             ("No calibrator, referral threshold, fusion model, outer OOF evaluation or reserved\ntest evaluation is implemented here.", "No calibrator, selected clinical referral threshold, fusion model or reserved\ntest evaluation is implemented by this report. The separate clean calibration\nstage remains distinct from these raw-score uncertainty diagnostics."),
             ("an explicitly selected recorded\nbest checkpoint", "the explicitly recorded matching\nouter-refit checkpoints"),
             ("on this split", "on the retained outer-held-out records"),
             ("All findings remain exploratory until confirmed within a frozen patient-safe\nevaluation procedure on independent outer patients.", "All findings remain conditional development diagnostics. The numerical recap\nuses saved clean results; an unchanged frozen procedure and the independently\nreserved test remain necessary before claiming final generalisation.")],
    })
    code = {
        3: _report_setup("11_image_uncertainty", "from melanoma_detection import dedup_notebook_images as image_docs\nimage_docs.original_style('uncertainty')\nuncertainty = image_docs.load_uncertainty(ROOT, dataset_identity)"),
        5: "image_docs.uncertainty_checks(uncertainty)",
        7: "if uncertainty['complete']:\n    display(pd.DataFrame(read_json(uncertainty['directory'] / 'inference_checks.json')['folds']))",
        9: "image_docs.uncertainty_views(ROOT, data, dataset_identity, finish_figure)",
        11: "display(pd.DataFrame([{'mode':'MC dropout', 'passes':30, 'seed':20261004, 'stochastic component':'trained explicit dropout only'}, {'mode':'TTA', 'passes':4, 'seed':20261004, 'stochastic component':'none; deterministic tensor flips'}]))",
        13: "image_docs.uncertainty_checks(uncertainty)",
        15: "image_docs.uncertainty_convergence(uncertainty, finish_figure)",
        17: "image_docs.uncertainty_quality(uncertainty, data, finish_figure)",
        19: "image_docs.uncertainty_distributions(uncertainty, finish_figure)",
        21: "if uncertainty['complete']:\n    display(pd.read_csv(uncertainty['directory'] / 'referral_rankings.csv'))",
        22: "image_docs.uncertainty_referral(uncertainty, finish_figure)",
        24: "if uncertainty['complete']:\n    display(pd.read_csv(uncertainty['directory'] / 'paired_patient_bootstrap_differences.csv'))",
        25: "image_docs.uncertainty_intervals(uncertainty, finish_figure)",
        27: "if uncertainty['complete']:\n    display(pd.DataFrame([{key: uncertainty['receipt'][key] for key in ('dataset_manifest_sha256','bootstrap_sampling','referral_score','test_set_loaded','calibration_fitted','cutoff_fitted')}]))",
        29: "image_docs.uncertainty_takeaways(uncertainty)",
    }
    return template_sections(name, code, md,
        context_note="**Genuine experiment-scope change:** clean matching outer-refit predictions\nreplace the historical one-inner-split selected checkpoint. Colours, referral\nlinestyles, two-panel convergence, five-bin reliability, histogram composition\nand MC-circle/TTA-square intervals retain the original visual language.")


def image_template_specs():
    return [
        ("09_image_modelling.ipynb", "09 · Clean B0 screening and nested evaluation", "Original pilot and nested-report templates, adapted to retained records.", modelling_sections()),
        ("10_bounded_image_pilots.ipynb", "10 · Bounded clean pilots and deferred transformations", "Original architecture, augmentation and lighting templates with explicit clean scopes.", pilot_sections()),
        ("11_image_uncertainty.ipynb", "11 · Clean image uncertainty", "Original uncertainty narrative and presentation on matching clean outer predictions.", uncertainty_sections()),
    ]
