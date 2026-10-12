import ast
import itertools
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

from melanoma_detection import metadata_tuning_v2_reporting as report

ROOT=Path(__file__).resolve().parents[1]


def predictions():
    rows=[]
    columns={name for _,a,b in report.FAMILIES for name in (a,b)}
    columns.update(f"p_nn_{prefix}seed_{seed}" for prefix in ("","tuned_") for seed in report.BASE_SEEDS)
    columns.update(["p_dummy","p_categorical_logistic"])
    for fold in range(5):
        for index in range(8):
            target=index%2
            row={"image_name":f"image_{fold}_{index}","patient_id":f"patient_{fold}_{index}","fold":fold,"target":target}
            for number,column in enumerate(sorted(columns)):
                row[column]=.1+.8*((index+number+fold)%8)/7
            row["p_dummy"]=.48+.005*fold
            rows.append(row)
    return pd.DataFrame(rows)


class MetadataROCReportTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_metrics_compute_mean_fold_roc_before_secondary_ap(self):
        oof=predictions()
        table=report.metric_table(oof,["p_spline"])
        self.assertLess(table.columns.get_loc("roc_auc"),table.columns.get_loc("average_precision"))
        primary=table[table.scope.eq("mean_fold")].iloc[0]
        self.assertAlmostEqual(primary.roc_auc,np.mean([roc_auc_score(g.target,g.p_spline) for _,g in oof.groupby("fold")]))
        self.assertAlmostEqual(primary.average_precision,np.mean([average_precision_score(g.target,g.p_spline) for _,g in oof.groupby("fold")]))

    def test_nn_plot_roc_left_original_colours_and_composition(self):
        oof=predictions();fig=report.plot_nn_comparison(oof)
        self.assertEqual(len(fig.axes),2)
        self.assertTrue(np.allclose(fig.get_size_inches(),[11,4.3]))
        self.assertEqual(fig.axes[0].get_ylabel(),"ROC-AUC")
        self.assertEqual(fig.axes[1].get_ylabel(),"Average precision")
        self.assertEqual([to_hex(p.get_facecolor()).upper() for p in fig.axes[0].patches],report.NN_COMPARISON_COLORS)
        expected=report.metric_table(oof,["p_spline",*[f"p_nn_seed_{s}" for s in report.BASE_SEEDS],"p_nn_mean"]).query("scope=='mean_fold'").roc_auc
        self.assertTrue(np.allclose([p.get_height() for p in fig.axes[0].patches],expected))

    def test_four_family_primary_bars_and_curve_grid(self):
        oof=predictions();fig=report.plot_tuning_bars(oof)
        colours=[to_hex(p.get_facecolor()).upper() for p in fig.axes[0].patches]
        self.assertEqual(colours,[report.BLUE]*4+[report.GOLD]*4)
        self.assertIn("primary",fig.axes[0].get_title())
        curves=report.plot_tuning_curves(oof)
        self.assertEqual(len(curves.axes),8)
        self.assertEqual(curves._suptitle.get_position()[1],1.002)

    def test_tuning_metrics_dispatch_displays_outer_fold_and_both_gain_figures(self):
        payload={"ready":True,"oof":predictions()}
        with patch.object(report,"_display_table") as tables, patch.object(report,"_show_figure") as figures:
            report.render_section(payload,"tuning_metrics")
        self.assertEqual(tables.call_count,2)
        scores=tables.call_args_list[0].args[0]
        changes=tables.call_args_list[1].args[0]
        self.assertLess(scores.columns.get_loc("roc_auc"),scores.columns.get_loc("average_precision"))
        self.assertEqual(list(changes.columns[:3]),["model","outer_fold","roc_auc_change"])
        self.assertEqual([call.args[2] for call in figures.call_args_list],
            ["four_family_fold_roc_auc_gains","four_family_fold_average_precision_gains"])
        for call in figures.call_args_list:
            self.assertEqual(len(call.args[1].axes),4)
            for ax in call.args[1].axes:
                self.assertTrue(np.array_equal(ax.lines[0].get_xdata(),np.arange(5)))

    def test_candidate_heatmap_uses_roc_scores_and_all_180_cells(self):
        rows=[]
        recipes=list(itertools.product([8,16,32,64],[.0003,.001,.003],[0,.2,.4]))
        for fold in range(5):
            for number,(width,lr,dropout) in enumerate(recipes):
                rows.append({"outer_fold":fold,"recipe_id":f"r{number:02}","recipe_order":number,
                    "hidden_units":width,"learning_rate":lr,"dropout":dropout,"selected":number==0,
                    "mean_inner_roc_auc":.5+.005*number+.001*fold,"mean_inner_average_precision":.03+.001*number})
        candidates=pd.DataFrame(rows);fig=report.plot_candidate_grid(candidates)
        values=np.concatenate([a.images[0].get_array().ravel() for a in fig.axes[:15]])
        self.assertEqual(len(values),180)
        self.assertTrue(np.allclose(sorted(values),sorted(candidates.mean_inner_roc_auc)))
        self.assertEqual(fig.axes[-1].get_ylabel(),"Mean inner-fold ensemble ROC-AUC")

    def test_baseline_curve_layout_matches_archived_original_for_both_conditions(self):
        oof=predictions()
        for condition in ["original","dedup_v1"]:
            data={"root":ROOT,"condition":condition,"stage":"05","oof":oof,"frame":oof}
            self.assertEqual(len(report.plot_baseline_curves(data).axes),8)

    def test_complete_notebook_depth_and_no_fitting_calls(self):
        prohibited={"fit","fit_transform","fit_final_bundle","run_tuning","evaluate_outer_fold","run_inference"}
        for condition in ["original","dedup_v1"]:
            for stage in ["04","05","06","07","08"]:
                nb=report.build_notebook(ROOT,condition,stage)
                metadata=nb.metadata.metadata_tuning_v2_report
                self.assertGreaterEqual(metadata.adapted_markdown_words,metadata.original_markdown_words)
                self.assertEqual(metadata.primary_metric,"mean_fold_roc_auc")
                for cell in nb.cells:
                    if cell.cell_type!="code":continue
                    source="\n".join(line for line in cell.source.splitlines() if not line.lstrip().startswith(("%","!")))
                    for node in ast.walk(ast.parse(source)):
                        if isinstance(node,ast.Call):
                            name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ""
                            self.assertNotIn(name,prohibited)

    def test_metric_priority_markdown_preserves_historical_ap_numbers(self):
        for condition in ["original","dedup_v1"]:
            for stage in ["04","05","06","07","08"]:
                book=report.build_notebook(ROOT,condition,stage)
                text="\n\n".join(c.source for c in book.cells if c.cell_type=="markdown")
                self.assertNotIn("mean-fold ROC-AUC is 0.0648",text)
                self.assertNotIn("Mean-fold ROC-AUC is 0.0458",text)
                self.assertNotIn("Average precision (AP), the primary measure",text)
                self.assertNotIn("Average precision (AP) is primary because",text)
                self.assertNotIn("The inner best AP",text)
                self.assertNotIn("mean outer-fold AP is the primary selection-procedure comparison",text)
                self.assertNotIn("Average precision (AP) is the primary ranking measure:",text)
                if stage=="05" and condition=="dedup_v1":
                    self.assertIn("historical mean-fold AP is 0.0648",text)
                    self.assertIn("Historical mean-fold AP is 0.0458",text)

    def test_all_robustness_writes_keep_the_v2_writer_after_late_imports(self):
        book=report.build_notebook(ROOT,"dedup_v1","08")
        binding=None;calls=0
        for cell in book.cells:
            if cell.cell_type!="code":continue
            for node in ast.parse(cell.source).body:
                if isinstance(node,ast.ImportFrom):
                    for alias in node.names:
                        if (alias.asname or alias.name)=="save_metadata_table":
                            binding=node.module
                            self.assertEqual(binding,"melanoma_detection.metadata_roc_historical")
                for call in ast.walk(node):
                    if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=="save_metadata_table":
                        self.assertEqual(binding,"melanoma_detection.metadata_roc_historical")
                        calls+=1
        self.assertGreaterEqual(calls,6)
        from melanoma_detection import dedup_notebook_metadata as legacy,metadata_roc_historical as current
        source=report._bind_v2_table_writer('from melanoma_detection.dedup_notebook_metadata import COLORS, save_metadata_table\nsave_metadata_table(ROOT, "08", "synthetic", TABLE)')
        table=pd.DataFrame({"value":[1]})
        with patch.object(legacy,"save_metadata_table",side_effect=AssertionError("Historical writer called")), patch.object(current,"save_metadata_table") as writer:
            exec(compile(source,"synthetic_writer_binding","exec"),{"ROOT":ROOT,"TABLE":table})
        writer.assert_called_once_with(ROOT,"08","synthetic",table)

    def test_robustness_appendix_retains_the_archived_level_two_hierarchy(self):
        for condition in ["original","dedup_v1"]:
            book=report.build_notebook(ROOT,condition,"08")
            headings=[line for cell in book.cells if cell.cell_type=="markdown"
                for line in cell.source.splitlines() if line.startswith("#")]
            self.assertEqual(headings[-1],"## Additional ROC-AUC-selected metadata experiment")
        # Other placements may nest the same assessment inside an existing section.
        self.assertTrue(report.metadata_nn_assessment_sections("original")[0][1].startswith("### Additional ROC-AUC-selected metadata experiment"))


if __name__=="__main__":
    unittest.main()
