"""
Unit tests for mpteval.dynamics
"""

import os
import partitura as pt

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from typing import Callable

from mpteval.dynamics import (
    midi_vel_to_rms,
    dynamic_range_in_db,
    get_upper_lower_stream_dynamic_range,
    dynamics_metrics_from_perf
)

DATA = os.path.dirname(os.path.abspath(__file__)) + "/data"
REF_MID = os.path.join(DATA, "ref.mid")
PRED_MID = os.path.join(DATA, "pred.mid")

EXPECTED_PRED  = {
    "dyn_corr" : 0.39562423493237187,
    "expected_rms_values" : [0.00520227, 0.06491231, 0.19779123, 0.39361755, 0.6680574,  1],
    "exptected_dynamic_range" : [21.92269103, 10.57213496, 23.7534581, 3.50372445, 45.67614913, 8.09851082],
    
}

class TestDynamicsMetrics(unittest.TestCase):

    def setUp(self):
        self.ref_perf = pt.load_performance_midi(REF_MID)
        self.pred_perf = pt.load_performance_midi(PRED_MID)
        self.ref_note_array = self.ref_perf.note_array()
        self.pred_note_array = self.pred_perf.note_array()
        self.dynamics_metrics = dynamics_metrics_from_perf(self.ref_perf, self.pred_perf)

    def test_midi_vel_to_rms(self):
        vel = np.array([0,25,51,76,102,127])
        rms = midi_vel_to_rms(vel)
        self.assertEqual(rms.shape, vel.shape)
        assert_array_almost_equal(rms, 
                                  EXPECTED_PRED["expected_rms_values"], 
                                  decimal=6, 
                                  err_msg=f"Expected RMS values {EXPECTED_PRED['expected_rms_values']} but got {rms}")
        
    def test_dynamic_range_in_db(self):
        vel1 = np.array([25, 102, 127, 127, 127, 127])
        vel2 = np.array([0, 51, 25, 102, 0, 76])
        range_db = dynamic_range_in_db(vel1, vel2)
        self.assertEqual(range_db.shape, vel1.shape)
        assert_array_almost_equal(range_db, 
                                  EXPECTED_PRED["exptected_dynamic_range"], 
                                  decimal=6, 
                                  err_msg=f"Expected dynamic range {EXPECTED_PRED['exptected_dynamic_range']} but got {range_db}")

    def test_get_upper_lower_stream_dynamic_range(self):
        dynamic_range_func, upper_voice_vel_func, lower_voice_vel_func, upper_voice, lower_voice = get_upper_lower_stream_dynamic_range(self.pred_note_array)

        self.assertIsInstance(dynamic_range_func, Callable)
        self.assertIsInstance(upper_voice_vel_func, Callable)
        self.assertIsInstance(lower_voice_vel_func, Callable)
        self.assertIsInstance(upper_voice, np.ndarray)
        self.assertIsInstance(lower_voice, np.ndarray)
        

    def test_dynamics_metrics_from_perf(self):
        corr = self.dynamics_metrics
        self.assertAlmostEqual(corr, 
                               EXPECTED_PRED["dyn_corr"], 
                               places=6, 
                               msg=f"Expected correlation {EXPECTED_PRED['dyn_corr']} but got {corr}")

if __name__ == '__main__':
    unittest.main()