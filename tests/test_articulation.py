"""
Unit tests for mpteval.articulation
"""

import os
import partitura as pt

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from typing import Callable

from mpteval.dynamics import PerformedChord
from mpteval.articulation import (
    chordify_perf_note_array,
    skyline_melody_identification,
    get_kor_stream_func,
    articulation_metrics_from_perf
)

DATA = os.path.dirname(os.path.abspath(__file__)) + "/data"
REF_MID = os.path.join(DATA, "ref.mid")
PRED_MID = os.path.join(DATA, "pred.mid")

EXPECTED_PRED ={
    "expected_note_array_lens" : [73, 38, 19, 1], # full note array, upper, lower, middle
    "expected_art_metrics_64" : (0.87665525, 0.29147666, 0.77280832),
    "expected_pedal_threshold_64" : 64,
}

class TestArticulationMetrics(unittest.TestCase):
    
    def setUp(self):
        self.ref_perf = pt.load_performance_midi(REF_MID)
        self.pred_perf = pt.load_performance_midi(PRED_MID)
        self.ref_note_array = self.ref_perf.note_array()
        self.pred_note_array = self.pred_perf.note_array()
        self.articulation_metrics = articulation_metrics_from_perf(self.ref_perf, self.pred_perf)

    def test_chordify_perf_note_array(self):
        result = chordify_perf_note_array(self.ref_note_array)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(chord, PerformedChord) for chord in result))

    def test_skyline_melody_identification(self):
        # test for type of return values for ref
        result = skyline_melody_identification(self.ref_note_array)
        self.assertEqual(len(result), 3)
        upper_voice, lower_voice, middle_voices = result    
        self.assertIsInstance(upper_voice, np.ndarray)
        self.assertIsInstance(lower_voice, np.ndarray)
        self.assertIsInstance(middle_voices, np.ndarray)
        
        # test length of return values for pred
        pred_upper, pred_lower, pred_middle = skyline_melody_identification(self.pred_note_array)
        self.assertEqual([len(self.pred_note_array), 
                          len(pred_upper),
                          len(pred_lower),
                          len(pred_middle)
                          ], 
                          EXPECTED_PRED['expected_note_array_lens'],
                          f"Expected {EXPECTED_PRED['expected_note_array_lens']} but got {[len(self.pred_note_array), len(pred_upper), len(pred_lower), len(pred_middle)]}")

    def test_get_kor_stream_func(self):
        kor_stream_func = get_kor_stream_func(self.ref_note_array)
        self.assertIsInstance(kor_stream_func, Callable)
        kor_values = kor_stream_func(self.ref_note_array['onset_sec'])
        self.assertIsInstance(kor_values, np.ndarray)

        
    def test_articulation_metrics_64(self):
        articulation_metrics_64 = list(self.articulation_metrics[0])
        *metrics_values, pedal_threshold = articulation_metrics_64
        assert_array_almost_equal(
            metrics_values, 
            EXPECTED_PRED['expected_art_metrics_64'],
            decimal=6, 
            err_msg=f"Expected {EXPECTED_PRED['expected_art_metrics_64']} but got {metrics_values}"
        )
        self.assertEqual(
            pedal_threshold, 
            EXPECTED_PRED['expected_pedal_threshold_64'],
            f"Expected {EXPECTED_PRED['expected_art_metrics_64']} but got {pedal_threshold}")

if __name__ == '__main__':
    unittest.main()