"""
Unit tests for mpteval.harmony
"""

import os
import partitura as pt
import numpy as np

import unittest
from typing import Callable

from mpteval.harmony import (
    get_tonal_tension_feature_func,
    harmony_metrics_from_perf
)

DATA = os.path.dirname(os.path.abspath(__file__)) + "/data"
REF_MID = os.path.join(DATA, "ref.mid")
PRED_MID = os.path.join(DATA, "pred.mid")

EXPECTED_PRED  = {
    "exptected_cloud_diameter_corr" : 0.68374109, 
    "exptected_cloud_momentum_corr" : 0.45588411,
}

class TestHarmonyMetrics(unittest.TestCase):

    def setUp(self):
        self.ref_perf = pt.load_performance_midi(REF_MID)
        self.pred_perf = pt.load_performance_midi(PRED_MID)
        self.ref_note_array = self.ref_perf.note_array()
        self.pred_note_array = self.pred_perf.note_array()
        self.harmony_metrics = harmony_metrics_from_perf(self.ref_perf, self.pred_perf)

    def test_get_tonal_tension_feature_func(self):
        cloud_diameter_vals, cloud_diameter_func = get_tonal_tension_feature_func(self.ref_note_array, 'cloud_diameter', ws=5, ss=1)       
        self.assertIsInstance(cloud_diameter_vals, np.ndarray)        
        self.assertIsInstance(cloud_diameter_func, Callable)        
        
        
    def test_harmony_metrics_from_perf(self):
        cloud_diameter_corr, cloud_momentum_corr = tuple(self.harmony_metrics[0])[:2]
        self.assertAlmostEqual(cloud_diameter_corr, 
                               EXPECTED_PRED["exptected_cloud_diameter_corr"], 
                               places=6, 
                               msg=f"Expected correlation {EXPECTED_PRED['exptected_cloud_diameter_corr']} but got {cloud_diameter_corr}")
        self.assertAlmostEqual(cloud_momentum_corr,
                                 EXPECTED_PRED["exptected_cloud_momentum_corr"],
                                 places=6,
                                 msg=f"Expected correlation {EXPECTED_PRED['exptected_cloud_momentum_corr']} but got {cloud_momentum_corr}")

if __name__ == '__main__':
    unittest.main()