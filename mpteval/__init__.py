#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Top-level module for mpteval"""
from importlib.metadata import version
from importlib.resources import files

# preprocessing and utils
from mpteval.core import config
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
from mpteval.utils_plotting import (
    plot_piano_roll_from_stream_note_arrays,
    plot_correlation
)

from mpteval.preprocessing import preprocess
from mpteval.alignment import dtw_align, dtw_dist
from mpteval.clustering import cluster, cluster_eval

from mpteval.metrics_feature import articulation, dynamics, harmony, timing
from mpteval.metrics import objective, majority

__version__ = version("mpteval")
REF_MID = str(files("mpteval").joinpath("assets/ref.mid"))
PRED_MID = str(files("mpteval").joinpath("assets/pred.mid"))