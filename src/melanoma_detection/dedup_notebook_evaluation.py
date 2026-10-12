"""Full clean model comparison with explicit conditional conclusions."""
from .dedup_notebook_data import section

COMPARISON_COLORS = {"b0_raw": "#235789", "spline_raw": "#2563EB", "metadata_nn_mean": "#333333"}
COMPARISON_LINES = {"b0_raw": "-", "spline_raw": "--", "metadata_nn_mean": "-."}
METRIC_LABELS = {"average_precision": "Average precision", "roc_auc": "ROC-AUC"}


def combined_extension_specs():
    return [("12_combined_evaluation.ipynb", "12 · Compare clean models and state the supported conclusions",
        "Combine matching clean outer predictions, probability checks, decision workload and patient uncertainty. The main comparison uses clean models against clean models; historical duplicate sensitivity remains supporting evidence.", [
        section("1. Establish completion and provenance before comparing", "A trained checkpoint alone is not a verified evaluation. Require dataset identity, matching image/patient/label/fold coverage, saved artifact fingerprints and reporting completion. Incomplete image stages remain pending and do not enter the final image-versus-metadata conclusion.", '''
            from melanoma_detection.dedup_notebook_evaluation import COMPARISON_COLORS, COMPARISON_LINES, METRIC_LABELS
            from melanoma_detection.dedup_notebook_metadata import LABELS
            COMPARISON_LABELS = {**LABELS, "b0_raw": "Raw B0"}
            evaluation = latest_evaluation(ROOT)
            bundle = score_bundle()
            expected = bundle["expected"]
            scores = bundle["scores"]
            display(pd.DataFrame([evaluation.get("stages", {})]))
            display(pd.DataFrame([{"evaluation_status":evaluation["status"], "matching_images":len(expected),
                "patients":expected.patient_id.nunique(),"malignant_images":int(expected.target.sum()),
                "dataset_manifest_sha256":dataset_identity["dataset_manifest_sha256"]}]))
            assert len(expected) == 6706 and expected.image_id.is_unique
            assert set(expected.image_id) == set(data.image_name)
            for name, values in scores.items():
                assert len(values) == len(expected) and np.isfinite(values).all()
            display(Markdown(evaluation.get("image_conclusion", "Clean comparison pending.")))
        '''),
        section("2. State the estimands and predeclared comparisons", "Primary ranking results are unweighted mean-fold AP and ROC-AUC: compute each held-out fold's metric, then average the five numbers. Pooled metrics rank all OOF rows together and can be affected by differences in score scales across folds, so they are secondary. AP summarizes precision as recall increases; ROC-AUC summarizes ranking across both classes. Accuracy is unsuitable for this rare-positive target. Historical predeclared contrasts are B0 minus fixed spline, B0 minus the prescribed fixed NN mean, and fixed NN mean minus spline. The separately labelled NN tuning appendix does not change those saved comparisons.", '''
            table = comparison_table()
            display(table)
            core_names=["b0_raw","spline_raw","metadata_nn_mean"]
            core=table.loc[table.model.isin(core_names)]
            fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
            for ax,metric,label in [(axes[0],"average_precision","Average precision"),(axes[1],"roc_auc","ROC-AUC")]:
                ax.bar([COMPARISON_LABELS[name] for name in core.model],core[metric],color=[COMPARISON_COLORS[name] for name in core.model])
                ax.set(title="Primary mean-fold "+label,ylabel=label); ax.tick_params(axis="x",rotation=15)
            finish_figure(fig,"core_mean_fold_ap_auc")
            if "b0_raw" not in scores: display(Markdown("**Pending:** B0 has not supplied all five verified outer predictions; current bars show metadata only."))
        '''),
        section("3. Read fold variation and pooled curves", "Replicate the original fold tables, ROC and precision–recall views on matching clean records. Curves here pool OOF scores and are secondary illustrations, while mean-fold metrics remain primary. A dashed prevalence line provides the constant-score reference for precision–recall. Fold variability can be large when only 24–26 malignant images occur in a group.", '''
            from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
            directory=project_path(ROOT,evaluation["output_directory"])
            metrics=pd.read_csv(directory/"metrics.csv")
            by_fold=metrics.loc[metrics.model.isin(core_names)&metrics.aggregation.eq("by_fold")&metrics.weighting.eq("image")]
            display(by_fold[["model","fold","n_images","n_malignant_images","n_malignant_patients","average_precision","roc_auc"]])
            fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
            for name in core_names:
                if name not in scores: continue
                fpr,tpr,_=roc_curve(expected.target,scores[name]); precision,recall,_=precision_recall_curve(expected.target,scores[name])
                axes[0].plot(fpr,tpr,color=COMPARISON_COLORS[name],linestyle=COMPARISON_LINES[name],label=f"{COMPARISON_LABELS[name]} (pooled AUC {roc_auc_score(expected.target,scores[name]):.3f})")
                axes[1].step(recall,precision,where="post",color=COMPARISON_COLORS[name],linestyle=COMPARISON_LINES[name],label=f"{COMPARISON_LABELS[name]} (pooled AP {average_precision_score(expected.target,scores[name]):.3f})")
            axes[0].plot([0,1],[0,1],"--",color="grey"); axes[1].axhline(expected.target.mean(),ls="--",color="grey",label="Retained prevalence")
            axes[0].set(title="Pooled clean OOF ROC (secondary)",xlabel="False-positive rate",ylabel="True-positive rate")
            axes[1].set(title="Pooled clean OOF precision–recall (secondary)",xlabel="Recall",ylabel="Precision")
            for ax in axes: ax.set_xlim(0,1); ax.set_ylim(0,1.02); ax.legend(fontsize=8)
            finish_figure(fig,"core_roc_precision_recall")
            fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
            for name,rows in by_fold.groupby("model"):
                for ax,metric in zip(axes,["average_precision","roc_auc"]): ax.plot(rows.fold.astype(int),rows[metric],marker="o",color=COMPARISON_COLORS[name],linestyle=COMPARISON_LINES[name],label=COMPARISON_LABELS[name])
            for ax,metric in zip(axes,["average_precision","roc_auc"]): ax.set(title="Outer-fold "+METRIC_LABELS[metric],xlabel="Preserved outer fold",ylabel=METRIC_LABELS[metric],xticks=range(5)); ax.legend(fontsize=8)
            finish_figure(fig,"core_fold_variation")
        '''),
        section("4. Use paired whole-patient intervals", "Sample patients with replacement within each outer fold for 2,000 draws, seed 42, and copy all their retained images together. Every model receives the same patient multiplicities on a draw. For a difference, compute A minus B on each draw, then use its 2.5th and 97.5th percentiles. Record invalid single-class draws. Intervals condition on these saved models and omit training/selection uncertainty; they do not establish clinical benefit.", '''
            intervals=pd.read_csv(directory/"bootstrap_intervals.csv")
            comparisons=pd.read_csv(directory/"paired_comparisons.csv")
            primary=comparisons.loc[comparisons.aggregation.eq("mean_fold")&comparisons.metric.isin(["average_precision","roc_auc"])]
            display(primary)
            display(intervals.loc[intervals.aggregation.eq("mean_fold")&intervals.metric.isin(["average_precision","roc_auc"])])
            fig,axes=plt.subplots(1,2,figsize=(13,4),layout="constrained")
            for ax,metric in zip(axes,["average_precision","roc_auc"]):
                selected=primary.loc[primary.metric.eq(metric)].reset_index(drop=True)
                pos=np.arange(len(selected))
                ax.hlines(pos,selected.ci_lower,selected.ci_upper,color="#235789",linewidth=2)
                ax.scatter(selected.difference,pos,color="#235789"); ax.axvline(0,color="grey",ls="--")
                comparison_labels = selected.comparison.copy()
                for name in core_names:
                    comparison_labels = comparison_labels.str.replace(name, COMPARISON_LABELS[name], regex=False)
                ax.set(yticks=pos,yticklabels=comparison_labels,xlabel="A minus B",title="Mean-fold "+METRIC_LABELS[metric]+": paired 95% intervals")
            finish_figure(fig,"core_paired_patient_intervals")
        '''),
        section("5. Separate patient weighting from patient-level prediction", "As a secondary diagnostic, give each image weight 1 divided by that patient's retained image count. This makes total image weight equal across patients while keeping the image-level target. It does not turn an image score into a patient diagnosis. Compare with conventional image-weighted results on the same records; never select a model merely because one weighting is favourable.", '''
            weighted=metrics.loc[metrics.model.isin(core_names)&metrics.aggregation.eq("mean_fold")]
            display(weighted[["model","weighting","average_precision","roc_auc"]])
            pivot=weighted.pivot(index="model",columns="weighting",values="average_precision")
            fig,ax=plt.subplots(figsize=(9,4),layout="constrained")
            pivot.rename(index=COMPARISON_LABELS).plot.bar(ax=ax,color=["#C47A14","#235789"],rot=15)
            ax.set(title="Mean-fold AP under two predeclared weightings",ylabel="Average precision")
            finish_figure(fig,"equal_patient_weight_ap")
        '''),
        section("6. Assess probabilities and training-only decision workload", "Ranking, calibration and decisions answer different questions. Brier and log loss assess probability quality; calibration maps and recall cutoffs must be learned inside each outer-training pool. Combine the resulting outer decisions without searching for a better cutoff on the reported OOF labels. Read recall beside precision and the number of images flagged. A fixed 0.5 cutoff is a diagnostic rather than the main recall scenario.", '''
            display(metrics.loc[metrics.aggregation.eq("mean_fold")&metrics.weighting.eq("image")&metrics.model.isin(
                ["b0_raw","spline_raw","metadata_nn_mean","spline_calibrated"]),["model","log_loss","brier"]])
            decisions=pd.read_csv(directory/"training_only_cutoff_metrics.csv")
            display(decisions)
            show_saved_table(evaluation["output_directory"]+"/fixed_cutoff_05.csv",12)
            selected=decisions.loc[decisions.fold.astype(str).eq("pooled")].copy()
            if not selected.empty:
                decision_labels = {("b0", "base_probability"): "Raw B0", ("b0", "calibrated_probability"): "Calibrated B0",
                    ("fixed_spline", "p_spline_raw"): LABELS["spline_raw"], ("fixed_spline", "p_spline_calibrated"): LABELS["spline_calibrated"]}
                labels = [decision_labels.get((row.model, row.score_scale), f"{row.model} / {row.score_scale}") for row in selected.itertuples()]
                fig,axes=plt.subplots(1,2,figsize=(13,4),layout="constrained")
                axes[0].barh(labels,selected.sensitivity,color="#235789"); axes[0].set(title="Training-only rule: pooled held-out recall",xlabel="Sensitivity",xlim=(0,1))
                axes[1].barh(labels,selected.flagged_fraction,color="#C47A14"); axes[1].set(title="Training-only rule: held-out review workload",xlabel="Flagged fraction",xlim=(0,1))
                finish_figure(fig,"training_only_recall_workload")
            if evaluation.get("stages",{}).get("image_calibration") != "complete": display(Markdown("**Pending:** clean image calibration and cutoffs are not yet complete. Metadata decisions above are the available training-only procedure."))
        '''),
        section("7. Inspect subgroup support before performance", "Recreate the age, sex, anatomical-site and known/unknown-site diagnostics. Subgroup results are pooled and descriptive; no subgroup-specific rule is tuned. Display image and malignant-patient counts together. A subgroup with only two malignant patients cannot justify a strong comparative conclusion, and undefined metrics remain missing rather than zero.", '''
            subgroups=pd.read_csv(directory/"subgroups.csv")
            selected=subgroups.loc[subgroups.model.isin(core_names)]
            display(selected)
            fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout="constrained")
            known=selected.loc[selected.axis.eq("site_knownness")]
            for ax,metric in zip(axes,["average_precision","roc_auc"]):
                for name,rows in known.groupby("model"):
                    ax.plot(rows.subgroup,rows[metric],marker="o",color=COMPARISON_COLORS[name],linestyle=COMPARISON_LINES[name],label=COMPARISON_LABELS[name])
                ax.set(title="Known/unknown site: descriptive "+METRIC_LABELS[metric],xlabel="Site availability",ylabel=METRIC_LABELS[metric]); ax.legend(fontsize=8)
            finish_figure(fig,"known_unknown_site_core")
            display(known[["subgroup","model","n_images","n_patients","n_malignant_images","n_malignant_patients"]])
        '''),
        section("8. Keep tuning, pilots and uncertainty in their proper roles", "Wider metadata searches and feature ablations are secondary. B0/ConvNeXt duration pilots use selected inner validation and cannot establish a final architecture winner. MC-dropout/TTA referral describes uncertainty and error ordering from matching clean checkpoints; its averages do not automatically replace calibrated decision scores. Stronger augmentation, CLAHE, resolution changes, full challenger OOF and fusion remain deferred.", '''
            secondary=table.loc[~table.model.isin(core_names)]
            display(secondary)
            statuses=[]
            for run in ["b0_duration_dedup_v1","convnext_duration_dedup_v1"]:
                _,record=pilot_report(run); statuses.append({"stage":run,"status":record.get("status","pending"),"evidence":"selected inner validation"})
            uncertainty=load_uncertainty_report(ROOT)
            statuses.append({"stage":"clean_uncertainty","status":uncertainty["status"],"evidence":"descriptive outer-held-out inference"})
            display(pd.DataFrame(statuses))
            display(Markdown("Read stages 07, 10 and 11 for their full diagnostics; these secondary observations cannot automatically change the locked primary comparison."))
        '''),
        section("9. Explain what duplicate sensitivity proves", "The saved historical prediction analysis removes copies from evaluation only. Clean retraining also removes their training and selection weight. The main report here compares clean models; the sensitivity appendix provides a traceable check without replacing the clean experiment. The reserved test and final all-development fit remain outside this analysis.", '''
            display(Markdown("[Saved-prediction sensitivity appendix](../../results/dedup_v1/sensitivity/saved_oof_v2/report.md). Retained historical scores were preserved exactly; they were not averaged or selected for favourable performance."))
            display(Markdown(evaluation.get("image_conclusion","Clean image comparison pending.")))
            display(Markdown("**Final reporting rule:** no image-versus-metadata conclusion until the required clean OOF and calibration stages are verified. Freeze the procedure before a separate final fit and any later reserved-test evaluation."))
        '''),
        section("10. Check reproducibility and hand off", "Keep manifests, predictions, histories, calibration maps and figures under separate clean identities. The notebooks provide editable plotting code and export PNG/PDF figures, while run receipts identify the fitting implementation and saved inputs. Historical files remain byte-identical. A pending stage is explicitly incomplete, not a negative result.", '''
            receipt_path=ROOT / DIRECTORY / "preservation_receipt.json"
            if receipt_path.exists(): display(pd.DataFrame([read_json(receipt_path)]))
            display(pd.DataFrame([{"saved_evaluation_directory":evaluation["output_directory"],"stage_figure_directory":str(FIGURE_DIR),
                "training_started_by_notebook":False,"reserved_test_loaded":False}]))
            display(Markdown("The clean procedure is ready for interpretation only to the extent of its verified completion. Keep original and clean outputs together when documenting limitations and provenance."))
        '''),
    ])]


def evaluation_specs():
    """Retain both original comparison narratives before combined extensions."""
    from .dedup_notebook_metadata import metadata_specs
    from .dedup_notebook_image_templates import nested_sections
    from .dedup_notebook_templates import MARKDOWN_CELL, CODE_CELL
    metadata = next(spec for spec in metadata_specs() if spec[0].startswith("08"))
    sections = []
    for kind, text, code in metadata[3]:
        if kind == MARKDOWN_CELL:
            text = text.replace("# Check metadata uncertainty, cutoffs and calibration", "# 12 — Combined clean evaluation: metadata uncertainty, cutoffs and calibration")
        sections.append((kind, text, code.replace("08_metadata_robustness", "12_combined_evaluation")))
    sections += [(MARKDOWN_CELL, "## Continue with the matching clean image evaluation\n\n"
        "The original metadata-robustness sections above establish probability, decision and "
        "subgroup interpretation. The original five-fold image report follows with the same "
        "retained image and patient identities. Only then does the combined comparison add "
        "paired model differences, equal-patient weighting and explicit completion checks. "
        "Saved clean fits are read throughout; no final all-development fit or reserved-test "
        "evaluation is performed here.", "")]
    sections += nested_sections()
    extension = combined_extension_specs()[0]
    sections += [(MARKDOWN_CELL, "## Combined evaluation extensions\n\n"
        "These additional checks document the deduplicated comparison beyond the original "
        "templates. Their primary summaries use unweighted mean-fold AP and ROC-AUC; "
        "pooled figures, patient weighting and subgroup comparisons remain secondary. "
        "Completion is read from verified experiment artifacts, not from a stale controller "
        "queue or historical pilot result.", "")]
    for heading, description, code in extension[3]:
        sections.append((MARKDOWN_CELL, f"### {heading}\n\n{description}", ""))
        sections.append((CODE_CELL, "", code))
    return [("12_combined_evaluation.ipynb", "12 · Combined clean model evaluation",
             "Original metadata-robustness and five-fold image templates with clean comparison extensions.", sections)]
