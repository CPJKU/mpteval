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

class TestMajorityCorrectPianoRoll(unittest.TestCase):

    def setUp(self):

        T = 300
        n_pitches = 5

        base   = np.zeros((n_pitches, T))
        roll_b = np.zeros((n_pitches, T))
        roll_c = np.zeros((n_pitches, T))

        # Pitch 0: false positive
        base[0, 5:40] = 64

        # Pitch 1: majority fill
        roll_b[1, 50:110] = 80
        roll_c[1, 50:110] = 60

        # Pitch 2: spurious short note
        base[2, 0:150]   = 64
        base[2, 160:180] = 64   # short blip, len=20 -> removed
        base[2, 200:280] = 64
        roll_b[2, 12:152] = 70
        roll_b[2, 165:185] = 64
        roll_b[2, 215:270] = 64
        roll_c[2, 5:160] = 64
        roll_c[2, 160:175] = 70
        roll_c[2, 222:267] = 65

        # Pitch 3: spurious re-onset / gap merge
        base[3, 0:120]   = 65   # left note
        base[3, 140:190] = 64   # middle note, gap of 20 before and after
        base[3, 210:270] = 67   # right note
        roll_b[3, 5:115]   = 62 
        roll_b[3, 138:195] = 62 
        roll_b[3, 208:268] = 62
        roll_c[3, 2:118]   = 66
        roll_c[3, 142:188] = 66
        roll_c[3, 212:272] = 66 

        # Pitch 4: true positive
        base[4, 50:150]   = 90
        roll_b[4, 50:151] = 70
        roll_c[4, 48:152] = 85  # slight jitter

        self.base   = csc_matrix(base)
        self.roll_b = csc_matrix(roll_b)
        self.roll_c = csc_matrix(roll_c)
        self.labels = ['base', 'roll_b', 'roll_c']
        self.T      = T

    def _get_result(self):
        with patch('librosa.load', return_value=(np.zeros(1000), 1000)), \
             patch('librosa.get_duration', return_value=self.T / 1000):
            result = majority_correct_piano_roll(
                self.base, self.roll_b, self.roll_c,
                audio_file='fake.wav',
                labels=self.labels
            )
        return np.array(result.todense(), dtype=float)

    def test_false_positive_deleted(self):
        """Single active frame agreed on by no other roll should be removed."""
        result = self._get_result()
        self.assertTrue(np.all(result[0] == 0),
            "Pitch 0 in base (false positive) should be deleted")

    def test_majority_fill_inserted(self):
        """Frames in pitch 1 active in roll_b and roll_c but not base should be inserted."""
        result = self._get_result()
        self.assertTrue(np.all(result[1, 50:110] == 70),
            "Pitch 1: majority-agreed frames t=50:110 should be filled in")

    def test_true_positive_kept(self):
        """Frames all rolls agree on in pitch 4 should be retained with base velocity."""
        result = self._get_result()
        self.assertTrue(np.all(result[4, 50:150] != 0),
            "Pitch 4: all-agreed note t=50:150 should be kept")
        self.assertTrue(np.all(result[4, 50:150] == 90),
            "Pitch 4: base velocity (90) should be kept")
        self.assertTrue(np.all(result[4, 151] == 0),
            "Pitch 4: offset jitter majority agreement should be cleaned")
            
    def test_spurious_short_note_removed(self):
        """Non-zero group in Pitch 2 shorter than min_length (50) should be zeroed after majority correction."""
        result = self._get_result()
        self.assertTrue(np.all(result[2, 160:180] == 0),
            "Pitch 2: short blip (len=20) at t=160:180 should be removed by smoothing")
        self.assertTrue(np.any(result[2, 0:150] != 0),
            "Pitch 2: long note t=0:150 should be retained")

    def test_reonset_gap_bridged_with_dominant_velocity(self):
        """Zero gap < merge_gap_threshold in pitch 3 between two long notes should be filled with the longer neighbour's value."""
        result = self._get_result()
        print(result[3, 0:260])
        self.assertTrue(np.all(result[3, 0:260] == 65),
            "Pitch 3: entire region t=0:260 should be one continuous note after merge")
        
if __name__ == '__main__':
    unittest.main()



