"""Check patient multiplicities and cutoff boundaries in notebook diagnostics."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from melanoma_detection.dedup_notebook_metadata import patient_draws, decision_intervals, exploratory_cutoff


class NotebookDiagnosticTests(unittest.TestCase):
    def test_patient_draws_keep_patient_images_and_fold_sample_counts(self):
        records=pd.DataFrame({"patient_id":["a","a","b","c","d","d"],"fold":[0,0,0,1,1,1]})
        draws,codes=patient_draws(records,repetitions=40,seed=42)
        self.assertTrue(np.array_equal(draws[:,0]+draws[:,1],np.full(40,2)))
        self.assertTrue(np.array_equal(draws[:,2]+draws[:,3],np.full(40,2)))
        self.assertEqual(codes[0],codes[1])
        self.assertEqual(codes[4],codes[5])
        again,_=patient_draws(records,repetitions=40,seed=42)
        self.assertTrue(np.array_equal(draws,again))

    def test_no_flags_have_undefined_precision_in_point_and_draws(self):
        records=pd.DataFrame({"patient_id":["a","b","c","d"],"fold":[0,0,1,1],"target":[1,0,1,0]})
        table=decision_intervals(records,np.zeros(4,dtype=bool),repetitions=40).set_index("metric")
        self.assertTrue(np.isnan(table.loc["precision","estimate"]))
        self.assertTrue(np.isnan(table.loc["precision","ci_lower"]))
        self.assertEqual(table.loc["precision","valid_draws"],0)
        self.assertEqual(table.loc["flagged_fraction","estimate"],0)

    def test_exploratory_cutoff_keeps_tied_scores_together(self):
        target=np.array([0,0,1,1]); scores=np.array([.1,.2,.2,.8])
        cutoff=exploratory_cutoff(target,scores)
        selected=scores>=cutoff
        self.assertEqual(selected[1],selected[2])
        self.assertIn(cutoff,np.unique(np.r_[0,scores,1]))


if __name__=="__main__": unittest.main()
