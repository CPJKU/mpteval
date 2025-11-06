#!/usr/bin/env python
"""Top-level module for mpteval"""
import pkg_resources

from . import articulation
from . import dynamics
from . import harmony
from . import timing
from . import benchmarks
from . import utils

__version__ = pkg_resources.get_distribution("mpteval").version

REF_MID = pkg_resources.resource_filename("mpteval", "assets/ref.mid")
PRED_MID = pkg_resources.resource_filename("mpteval", "assets/pred.mid")