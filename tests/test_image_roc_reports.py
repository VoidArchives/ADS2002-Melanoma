"""Synthetic checks for read-only image presentation and ROC/AP meanings."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from melanoma_detection import image_roc_reporting as reporting
spec = importlib.util.spec_from_file_location("image_roc_builder",ROOT / "scripts/build_image_roc_reports.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def synthetic_outer():
    rows=[]
    for fold in range(5):
        y=np.array([0,0,0,1,1]);p=np.array([.05,.4,.75,.3,.8])+fold*.002
        for index,(target,score) in enumerate(zip(y,p)):
            rows.append({"image_id":f"f{fold}_{index}","patient_id":f"p{fold}_{index}","fold":fold,"target":target,"base_probability":score})
    frame=pd.DataFrame(rows)
    metrics=pd.DataFrame([{"fold":f,"roc_auc":roc_auc_score(r.target,r.base_probability),"average_precision":average_precision_score(r.target,r.base_probability),"prevalence":r.target.mean()} for f,r in frame.groupby("fold")])
    return {"complete":True,"predictions":frame,"fold_metrics":metrics}


class ImageRocTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_all_eleven_use_verified_full_templates_and_read_only_code(self):
        for relative in builder.NOTEBOOKS:
            with self.subTest(notebook=relative):
                nb=builder.build_notebook(ROOT,relative)
                metadata=nb.metadata.image_roc_report
                self.assertGreaterEqual(metadata.markdown_words,.9*metadata.template_markdown_words)
                self.assertEqual(metadata.primary_metric,"mean_fold_roc_auc")
                self.assertFalse(metadata.image_compute_on_execute)
                self.assertTrue(all(c.execution_count is None and not c.outputs for c in nb.cells if c.cell_type=="code"))
                code="\n".join(c.source for c in nb.cells if c.cell_type=="code")
                self.assertNotIn('run_smoke(',code)
                self.assertNotIn('run_inference(',code)
                self.assertNotIn('load_selected_checkpoint(',code)
                if "Subset_dedup/" in relative:
                    self.assertNotIn('FIGURE_DIR = ROOT / DIRECTORY',code)
                    self.assertIn('report_directory(ROOT, "dedup_v1", NOTEBOOK_STAGE)',code)
                if relative.endswith(("14_image_nested_evaluation.ipynb","12_combined_evaluation.ipynb")):
                    self.assertIn('run_id="roc_auc_grid36_v1", experiment_version=2',code)

    def test_fold_plot_uses_actual_roc_auc_and_preserves_original_composition(self):
        raw=synthetic_outer();figures=[]
        with patch.object(reporting,"display"):
            reporting.raw_outer_ranking(raw,lambda fig,name:figures.append((name,fig)))
        self.assertEqual([n for n,f in figures],["outer_fold_roc_auc","outer_fold_roc_pr"])
        scatter=figures[0][1].axes[0]
        self.assertEqual(tuple(figures[0][1].get_size_inches()),(8,4.5))
        observed=[float(c.get_offsets()[0,1]) for c in scatter.collections[:5]]
        np.testing.assert_allclose(observed,raw["fold_metrics"].roc_auc)
        self.assertFalse(np.allclose(observed,raw["fold_metrics"].average_precision))
        self.assertAlmostEqual(scatter.lines[0].get_ydata()[0],raw["fold_metrics"].roc_auc.mean())
        self.assertEqual(len(figures[1][1].axes),2)
        self.assertEqual(tuple(figures[1][1].get_size_inches()),(12,4.7))
        self.assertEqual(figures[1][1].axes[0].get_title(),"Raw OOF ROC by outer fold")

    def test_sensitivity_caption_and_two_panels_do_not_relabel_ap_numbers(self):
        nb=builder.build_notebook(ROOT,"notebooks/Subset/15_deduplicated_workflow.ipynb")
        rows=[]
        for population,auc,ap in [("original",.82,.14),("retained",.79,.12)]:
            for model in ("b0_raw","spline_raw","metadata_nn_mean"):
                rows.append({"population":population,"model":model,"aggregation":"mean_fold","weighting":"image","roc_auc":auc,"average_precision":ap,"log_loss":.2,"brier":.1})
        output=[];namespace={"pd":pd,"np":np,"plt":plt,"sensitivity_folder":ROOT,"display":output.append,"LABELS":{"b0_raw":"B0","spline_raw":"Spline","metadata_nn_mean":"NN"},"COLOURS":{"b0_raw":"#2563EB","spline_raw":"#C78513","metadata_nn_mean":"#505963"}}
        from IPython.display import Markdown
        namespace["Markdown"]=Markdown
        with patch.object(pd,"read_csv",return_value=pd.DataFrame(rows)),patch.object(plt,"show"):
            exec(nb.cells[12].source,namespace)
        texts=[v.data for v in output if isinstance(v,Markdown)]
        self.assertIn("0.820000",texts[-1]);self.assertIn("0.790000",texts[-1])
        axes=plt.gcf().axes
        self.assertIn("ROC-AUC",axes[0].get_title());self.assertIn("AP",axes[1].get_title())
        self.assertAlmostEqual(axes[0].patches[0].get_height(),.82)
        self.assertAlmostEqual(axes[1].patches[0].get_height(),.14)

    def test_entire_original_uncertainty_notebook_reads_saved_tables_and_six_figures(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder=Path(directory)
            for name in ("smoke_receipt.json","raw_prediction_checks.json","inference_checks.json","inference_provenance.json","preservation_receipt.json"):
                (folder/name).write_text(json.dumps({"verified":True}),encoding="utf-8")
            summary=pd.DataFrame({"image_id":["a","b"],"patient_id":["pa","pb"],"target":[0,1],"base_score":[.1,.9],"mc_mean_score":[.15,.85],"mc_score_sd":[.02,.03],"tta_mean_score":[.12,.88],"tta_score_sd":[.01,.04]})
            quality=pd.DataFrame([{"prediction":name,"roc_auc":.88,"average_precision":.42,"log_loss":.2,"brier":.1} for name in ("Deterministic","MC mean","TTA mean","Constant fitting prevalence")])
            referral=pd.DataFrame([{"method":method,"referral_fraction":.2,"ranking":ranking,"n_malignant_images_referred":1} for method in ("mc","tta") for ranking in ("uncertainty_sd","prediction_score","random")])
            differences=pd.DataFrame([{"method":method,"referral_fraction":.2,"comparator":"prediction_score","metric":metric,"estimate":.01,"ci_lower":-.02,"ci_upper":.03} for method in ("mc","tta") for metric in ("malignant_referred_fraction","retained_error")])
            stored={"directory":folder,"provenance":{"source_run":"b0_augmentation_12e_v1","architecture":"efficientnet_b0","recipe_id":"lr3e5_sqrt"},"checkpoint":{"best_epoch":6,"fitting_patient_ids":["fit"],"evaluation_patient_ids":["pa","pb"],"outer_excluded_patient_ids":["out"],"checkpoint_sha256":"saved","evaluation_status":"exploratory_inner_validation"},"summary":summary,"quality":quality,"convergence":pd.DataFrame({"n_passes":[5,10,20]}),"reliability":pd.DataFrame({"n_images":[2]}),"referral":referral,"differences":differences}
            nb=builder.build_notebook(ROOT,"notebooks/Subset/13_image_uncertainty.ipynb")
            figures=[]
            with patch.object(reporting,"load_saved_original_uncertainty",return_value=stored),patch.object(reporting,"report_directory",return_value=folder),patch.object(reporting,"show_saved_figure",side_effect=lambda root,path:figures.append(path)),patch("IPython.display.display"):
                namespace={}
                for cell in nb.cells:
                    if cell.cell_type=="code":exec(cell.source,namespace)
            self.assertEqual(len(figures),6)
            self.assertTrue(all(reporting.UNCERTAINTY_RUN in p for p in figures))
            self.assertEqual(set(p.name for p in folder.iterdir()),{"smoke_receipt.json","raw_prediction_checks.json","inference_checks.json","inference_provenance.json","preservation_receipt.json"})

    def test_exports_reject_historical_paths_before_writing(self):
        figure=plt.figure()
        with self.assertRaisesRegex(ValueError,"v2 report namespace"):
            reporting.save_figure(figure,ROOT / "results/image_d1_runs/b0_pilot_v1/figures/forbidden.png")
        for stage in ("../escape","nested/name",""):
            with self.assertRaises(ValueError):reporting.report_directory(ROOT,"original",stage)

    def test_pilot_bars_report_roc_without_changing_historical_ap_epoch_role(self):
        rows=[{"family":"B0","cap":cap,"selection_kind":kind,"roc_auc":.81 if kind=="endpoint" else .77,"average_precision":.14,"log_loss":.2,"brier":.1} for kind in ("endpoint","best_in_cap") for cap in (12,24)]
        figures=[]
        with patch.object(reporting,"display"):
            reporting.pilot_endpoint_metrics({"complete":True,"metrics":pd.DataFrame(rows)},lambda fig,name:figures.append(fig))
        self.assertEqual(tuple(figures[0].get_size_inches()),(11,4.4))
        self.assertAlmostEqual(figures[0].axes[0].patches[0].get_width(),.81)
        self.assertAlmostEqual(figures[0].axes[1].patches[0].get_width(),.77)
        self.assertIn("Historical AP-selected",figures[0].axes[1].get_title())
        self.assertIn("ROC-AUC",figures[0].axes[0].get_xlabel())


if __name__=="__main__":unittest.main()
