"""
Backbone registry for the controlled robustness study.

Only the Xception backbone required by Xception, UCF, and SPSL is
registered in the study runtime. Other DeepfakeBench backbone
implementations remain in the repository for upstream provenance.
"""

import os
import sys


current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(
    os.path.dirname(current_file_path)
)
project_root_dir = os.path.dirname(
    parent_dir
)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)


from metrics.registry import BACKBONE

from .xception import Xception


__all__ = [
    "BACKBONE",
    "Xception",
]