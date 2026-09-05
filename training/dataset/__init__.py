"""
Dataset registry for the controlled robustness study.

The study runtime requires only:

* DeepfakeAbstractBaseDataset
    for Xception, SPSL, and DEV/evaluation loading;

* pairDataset
    for the detector-specific UCF training mechanism.

Other DeepfakeBench dataset implementations remain in the repository for
upstream provenance but are not imported into the study runtime.
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


from .abstract_dataset import DeepfakeAbstractBaseDataset
from .pair_dataset import pairDataset


__all__ = [
    "DeepfakeAbstractBaseDataset",
    "pairDataset",
]