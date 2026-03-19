#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Top-level module for mpteval"""
import pkg_resources

# preprocessing and utils
from mpteval.core import config
from mpteval import utils
from mpteval.utils import (
    is_monophonic,
    midi_to_piano_note,
    plot_piano_roll,
    create_note_list,
    alignment_list_to_df,
    group_by_structure,
    pairwise_distance_matrix,
    accumulated_cost_matrix,
    optimal_warping_path,
    dynamic_time_warping,
    fast_dynamic_time_warping,
    greedy_note_alignment,
    notewise_alignment,
)

from mpteval.preprocessing import preprocess
from mpteval.alignment import dtw_align, dtw_dist
from mpteval.clustering import cluster, cluster_eval

from mpteval.features import articulation, dynamics, harmony, timing
from mpteval.metrics import objective, subjective

__version__ = pkg_resources.get_distribution("mpteval").version

REF_MID = pkg_resources.resource_filename("mpteval", "assets/ref.mid")
PRED_MID = pkg_resources.resource_filename("mpteval", "assets/pred.mid")