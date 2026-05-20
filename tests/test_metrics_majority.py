"""
Unit tests for mpteval.harmony
"""

import os

import partitura as pt
import numpy as np
from scipy.sparse import csc_matrix

import unittest
from unittest.mock import patch

from typing import Callable

from mpteval.metrics.majority import (
    majority_correct_piano_roll,
    compare_piano_rolls_labeled,
    compare_piano_rolls
)

class TestMajorityMetrics(unittest.TestCase):

    def setUp(self):
        # Piano roll shape: (3 pitches, ~10 time steps)

        # Pitch 0: base has a note at t=2 -> only base has it -> DELETED (false positive)
        # Pitch 1: base is missing a note at t=5 -> both roll_b and roll_c have it -> INSERTED (false negative)
        # Pitch 2: all three agree at t=8 -> should be KEPT as-is

        #         t: 0  1  2  3  4  5  6  7  8  9
        # base  p0:  0  0  64 0  0  0  0  0  0  0   <- only base active at t=2, should be deleted
        # base  p1:  0  0  0  0  0  0  0  0  0  0   <- base missing at t=5, b and c have it, should be filled
        # base  p2:  0  0  0  0  0  0  0  0  90 0   <- all agree, should stay

        # roll_b p0: 0  0  0  0  0  0  0  0  0  0
        # roll_b p1: 0  0  0  0  0  80 0  0  0  0
        # roll_b p2: 0  0  0  0  0  0  0  0  70 0

        # roll_c p0: 0  0  0  0  0  0  0  0  0  0
        # roll_c p1: 0  0  0  0  0  60 0  0  0  0
        # roll_c p2: 0  0  0  0  0  0  0  0  85 0

        n_pitches, n_base, n_b, n_c = 3, 10, 9, 12

        base = np.zeros((n_pitches, n_base))
        base[0, 2] = 64   # minority
        base[2, 8] = 90   # true positive

        roll_b = np.zeros((n_pitches, n_b))
        roll_b[1, 5] = 80   # majority
        roll_b[2, 8] = 70   # true positive

        roll_c = np.zeros((n_pitches, n_c))
        roll_c[1, 5] = 60   # majority
        roll_c[2, 8] = 85   # true positive

        self.base   = csc_matrix(base)
        self.roll_b = csc_matrix(roll_b)
        self.roll_c = csc_matrix(roll_c)
        self.labels = ['base', 'roll_b', 'roll_c']

    def test_majority_correct_piano_roll(self):
        with patch('librosa.load', return_value=(np.zeros(1000), 1000)), \
             patch('librosa.get_duration', return_value=0.01):  # 10ms
            result = majority_correct_piano_roll(
                self.base, self.roll_b, self.roll_c,
                audio_file='fake.wav',
                labels=self.labels
            )

        result_dense = np.array(result.todense(), dtype=float)

        # False positive: base had a lone note at pitch=0, t=2 -> should be zeroed
        self.assertEqual(result_dense[0, 2], 0,
            "False positive at pitch=0, t=2 should be deleted")

        # Omission: base missed pitch=1 at t=5, roll_b=80 and roll_c=60 -> mean=70
        expected_fill = np.round((80 + 60) / 2)
        self.assertEqual(result_dense[1, 5], expected_fill,
            f"Omission at pitch=1, t=5 should be filled with mean velocity {expected_fill}")

        # True positive: all agree at pitch=2, t=8 -> base value preserved
        self.assertEqual(result_dense[2, 8], 90,
            "Agreed note at pitch=2, t=8 should be kept")
        
if __name__ == '__main__':
    unittest.main()



