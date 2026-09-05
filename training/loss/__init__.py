"""
Loss registry for the controlled robustness study.

Required study losses:

* cross_entropy:
    Xception and SPSL classification;
    UCF common and manipulation-specific classification.

* contrastive_regularization:
    UCF representation regularization.

* l1loss:
    UCF reconstruction objective.

Other DeepfakeBench losses remain in the repository but are intentionally
not registered in the study runtime.
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


from metrics.registry import LOSSFUNC

from .cross_entropy_loss import CrossEntropyLoss
from .contrastive_regularization import ContrastiveLoss
from .l1_loss import L1Loss


__all__ = [
    "LOSSFUNC",
    "CrossEntropyLoss",
    "ContrastiveLoss",
    "L1Loss",
]