#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Top-level module for mpteval"""
import pkg_resources

# preprocessing and utils
from . import config
from . import preprocess

# alignment
from . import dtw_align
from . import dtw_dist

# clustering
from . import cluster
from . import cluster_eval

# information retrieval (objective) and perceptually motivated (subjective) metrics
from . import objective
from . import subjective

# performance metrics
from . import articulation
from . import dynamics
from . import harmony
from . import timing

from . import utils

__version__ = pkg_resources.get_distribution("mpteval").version

REF_MID = pkg_resources.resource_filename("mpteval", "assets/ref.mid")
PRED_MID = pkg_resources.resource_filename("mpteval", "assets/pred.mid")