"""
Detector registry for the controlled robustness study.

Only the three detectors used by the study are imported and registered:

* Xception
* UCF
* SPSL

The remaining DeepfakeBench detector implementations are retained in the
repository for upstream provenance but are intentionally not imported into
the study runtime.
"""

import os
import sys


current_file_path = os.path.abspath(
    __file__
)

parent_dir = os.path.dirname(
    os.path.dirname(
        current_file_path
    )
)

project_root_dir = os.path.dirname(
    parent_dir
)

if parent_dir not in sys.path:
    sys.path.append(
        parent_dir
    )

if project_root_dir not in sys.path:
    sys.path.append(
        project_root_dir
    )


from metrics.registry import DETECTOR

from .xception_detector import XceptionDetector
from .ucf_detector import UCFDetector
from .spsl_detector import SpslDetector


__all__ = [
    "DETECTOR",
    "XceptionDetector",
    "UCFDetector",
    "SpslDetector",
]