#!/usr/bin/env python
"""Top-level module for mpteval"""

# Import all submodules (for each task)
from . import articulation
from . import dynamics
from . import harmony
from . import timing
from . import util

__version__ = "0.1.0"

import pkg_resources

REF_MID = pkg_resources.resource_filename("mpteval", "assets/ref.mid")
PRED_MID = pkg_resources.resource_filename("mpteval", "assets/pred.mid")