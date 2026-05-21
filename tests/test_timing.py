"""
Unit tests for mpteval.timing
"""

import os
import partitura as pt

import unittest
from typing import Callable

from mpteval.metrics_feature.timing import (
    get_ioi_stream_func,
    timing_metrics_from_perf
)

DATA = os.path.dirname(os.path.abspath(__file__)) + "/data"
REF_MID = os.path.join(DATA, "ref.mid")
PRED_MID = os.path.join(DATA, "pred.mid")

EXPECTED_PRED  = {
    "expected_stream_lens" : [22, 44],
    "exptected_melody_ioi_corr" : 0.7579011, 
    "exptected_bass_ioi_corr" : 0.3159058,
}

class TestTimingMetrics(unittest.TestCase):

    def setUp(self):
        self.ref_perf = pt.load_performance_midi(REF_MID)
        self.pred_perf = pt.load_performance_midi(PRED_MID)
        self.ref_note_array = self.ref_perf.note_array()
        self.pred_note_array = self.pred_perf.note_array()
        self.timing_metrics = timing_metrics_from_perf(self.ref_perf, self.pred_perf)

    def test_get_ioi_stream_func(self):
        ioi_stream_func = get_ioi_stream_func(self.ref_note_array)
        self.assertIsInstance(ioi_stream_func, Callable)        
        
    def test_timing_metrics_from_perf(self):
        melody_ioi_corr, bass_ioi_corr = tuple(self.timing_metrics[0])[:2]
        self.assertAlmostEqual(melody_ioi_corr, 
                               EXPECTED_PRED["exptected_melody_ioi_corr"], 
                               places=6, 
                               msg=f"Expected correlation {EXPECTED_PRED['exptected_melody_ioi_corr']} but got {melody_ioi_corr}")
        self.assertAlmostEqual(bass_ioi_corr,
                                 EXPECTED_PRED["exptected_bass_ioi_corr"],
                                 places=6,
                                 msg=f"Expected correlation {EXPECTED_PRED['exptected_bass_ioi_corr']} but got {bass_ioi_corr}")

if __name__ == '__main__':
    unittest.main()