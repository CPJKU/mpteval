"""
Unit tests for mpteval.timing
"""

import os
import partitura as pt

import unittest
from typing import Callable

from mpteval.timing import (
    get_ioi_stream_func,
    skyline_melody_identification_from_array,
    timing_metrics_from_perf
)

DATA = os.path.dirname(os.path.abspath(__file__)) + "/data"
REF_MID = os.path.join(DATA, "ref.mid")
PRED_MID = os.path.join(DATA, "pred.mid")

EXPECTED_PRED  = {
    "expected_stream_lens" : [28, 38],    
    "exptected_melody_ioi_corr" : 0.13976843, 
    "exptected_acc_ioi_corr" : 0.5453803,
}

class TestTimingMetrics(unittest.TestCase):

    def setUp(self):
        self.ref_pref = pt.load_performance_midi(REF_MID)
        self.pred_perf = pt.load_performance_midi(PRED_MID)
        self.ref_note_array = self.ref_pref.note_array()
        self.pred_note_array = self.pred_perf.note_array()
        self.timing_metrics = timing_metrics_from_perf(self.ref_pref, self.pred_perf)

    def test_skyline_melody_identification_from_array(self):
        melody, accompaniment = skyline_melody_identification_from_array(self.ref_note_array)
        streams_lens = [len(melody), len(accompaniment)]
        self.assertEqual(streams_lens, EXPECTED_PRED["expected_stream_lens"], f"Expected stream lengths {EXPECTED_PRED['expected_stream_lens']} but got {streams_lens}")
    
    def test_get_ioi_stream_func(self):
        ioi_stream_func = get_ioi_stream_func(self.ref_note_array)
        self.assertIsInstance(ioi_stream_func, Callable)        
        
    def test_timing_metrics_from_perf(self):
        melody_ioi_corr, acc_ioi_corr = tuple(self.timing_metrics[0])[:2]
        self.assertAlmostEqual(melody_ioi_corr, 
                               EXPECTED_PRED["exptected_melody_ioi_corr"], 
                               places=6, 
                               msg=f"Expected correlation {EXPECTED_PRED['exptected_melody_ioi_corr']} but got {melody_ioi_corr}")
        self.assertAlmostEqual(acc_ioi_corr,
                                 EXPECTED_PRED["exptected_acc_ioi_corr"],
                                 places=6,
                                 msg=f"Expected correlation {EXPECTED_PRED['exptected_acc_ioi_corr']} but got {acc_ioi_corr}")

if __name__ == '__main__':
    unittest.main()