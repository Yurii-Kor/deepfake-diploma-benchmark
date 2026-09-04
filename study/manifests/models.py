from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple


DATASET_FFPP = "FaceForensics++"
DATASET_CELEB_DF_V2 = "Celeb-DF-v2"


SOURCE_SPLIT_TRAIN = "train"
SOURCE_SPLIT_VAL = "val"
SOURCE_SPLIT_TEST = "test"

VALID_SOURCE_SPLITS = (
    SOURCE_SPLIT_TRAIN,
    SOURCE_SPLIT_VAL,
    SOURCE_SPLIT_TEST,
)


ROLE_FIT = "fit"
ROLE_DEVELOPMENT = "development"
ROLE_VALIDATION = "validation"
ROLE_TEST = "test"
ROLE_EXTERNAL_EVALUATION = "external_evaluation"

VALID_ROLES = (
    ROLE_FIT,
    ROLE_DEVELOPMENT,
    ROLE_VALIDATION,
    ROLE_TEST,
    ROLE_EXTERNAL_EVALUATION,
)


LABEL_REAL = 0
LABEL_FAKE = 1

VALID_LABELS = (
    LABEL_REAL,
    LABEL_FAKE,
)


FFPP_MANIPULATIONS = (
    "DeepFakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
)


SourcePair = Tuple[str, str]


def canonical_source_group_id(
    first_source_id: str,
    second_source_id: str,
) -> str:
    """
    Return a direction-independent identifier for one FF++ source pair.

    FF++ manipulated samples can exist in both directions, for example
    ``156_243`` and ``243_156``. Both must belong to the same study-level
    source group so that fitting/development separation cannot be violated.
    """
    first = str(first_source_id).strip()
    second = str(second_source_id).strip()

    if not first:
        raise ValueError("First source ID must not be empty.")

    if not second:
        raise ValueError("Second source ID must not be empty.")

    if first == second:
        raise ValueError(
            "FF++ source pair must contain two different source IDs: {}."
            .format(first)
        )

    ordered = sorted(
        (
            first,
            second,
        )
    )

    return "{}_{}".format(
        ordered[0],
        ordered[1],
    )


@dataclass(frozen=True)
class StudyVideoRecord:
    """
    Canonical study-level identity and role for one source video.

    This record deliberately contains no frame sampling, face geometry,
    detector outputs, or processing-condition information. Those belong to
    later pipeline stages.

    ``source_split`` describes the dataset-native split.

    ``role`` describes the experimental role assigned by this study.

    ``source_group_id`` links all source-related FF++ videos that must remain
    together when the official training partition is subdivided into fitting
    and development subsets.

    ``source_video_id`` and ``target_video_id`` retain directional source
    information where it exists. For an FF++ manipulated video such as
    ``156_243``, the two fields are ``156`` and ``243`` respectively. For
    pristine videos and datasets without this directional relation,
    ``target_video_id`` is ``None``.
    """

    dataset: str
    source_split: str
    role: str

    base_video_id: str
    source_group_id: str

    source_video_id: str
    target_video_id: Optional[str]

    study_label: int
    manipulation: Optional[str]

    relative_source_path: str
    source_label: Optional[str] = None


def record_to_dict(
    record: StudyVideoRecord,
) -> Dict[str, object]:
    """
    Convert a canonical record to a serialization-friendly dictionary.
    """
    return asdict(record)