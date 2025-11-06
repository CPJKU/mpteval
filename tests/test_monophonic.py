'''
Current test cases handle case where both reference and prediction are monophonic, and test only time-related metrics as only those are computed for separate streams.
'''

import os
import partitura as pt
import numpy as np

import unittest
from typing import Callable

import partitura as pt

from mpteval.articulation import articulation_metrics_from_perf
from mpteval.timing import timing_metrics_from_perf
from mpteval.harmony import harmony_metrics_from_perf
from mpteval.dynamics import dynamics_metrics_from_perf

DATA = os.path.dirname(os.path.abspath(__file__)) + "/data"
REF_MID = os.path.join(DATA, "monophonic.mid")
PRED_MID = os.path.join(DATA, "monophonic.mid") # we measure monophonic input against itself

EXPECTED_PRED  = {
    "exptected_timing_metrics_monophonic" : (1., np.nan, np.nan), 
    "exptected_articulation_metrics_monophonic" : (1., np.nan, np.nan,  64), 
    "exptected_harmony_metrics_from_perf" : (1., 1., 1.), 
    "exptected_dynamics_metrics_from_perf" : 1.0, 
}

class TestMetricsMonophonic(unittest.TestCase):

    def setUp(self):
        self.ref_perf = pt.load_performance_midi(REF_MID)
        self.pred_perf = pt.load_performance_midi(PRED_MID)

    def test_timing_metrics_both_monophonic(self):
        timing_metrics = timing_metrics_from_perf(self.ref_perf, self.pred_perf)[0]
        exp_melody_corr = EXPECTED_PRED['exptected_timing_metrics_monophonic'][0]
        self.assertAlmostEqual(timing_metrics[0], exp_melody_corr, msg=f'Expected correlation {exp_melody_corr}, but got timing_metrics[0].')
        self.assertTrue(np.isnan(timing_metrics[1]), msg=f'Given monophonic input expected undefined (nan) return for accompaniment metric, but got {timing_metrics[1]}')
        self.assertTrue(np.isnan(timing_metrics[2]), msg=f'Given monophonic input expected undefined (nan) return for melody-accompaniment ratio metric, but got {timing_metrics[2]}')
        
    def test_articulation_metrics_both_monophonic(self):
        articulation_metrics = articulation_metrics_from_perf(self.ref_perf, self.pred_perf)[0] # [0] = check for pedal threshold 64
        exp_melody_corr = EXPECTED_PRED['exptected_articulation_metrics_monophonic'][0]
        self.assertAlmostEqual(articulation_metrics[0], exp_melody_corr, msg=f'Expected correlation {exp_melody_corr}, but got timing_metrics[0].')
        self.assertTrue(np.isnan(articulation_metrics[1]), msg=f'Given monophonic input expected undefined (nan) return for accompaniment metric, but got {articulation_metrics[1]}')
        self.assertTrue(np.isnan(articulation_metrics[2]), msg=f'Given monophonic input expected undefined (nan) return for melody-accompaniment ratio metric, but got {articulation_metrics[2]}')
        
if __name__ == '__main__':
    unittest.main()