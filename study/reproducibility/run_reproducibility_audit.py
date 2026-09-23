from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from study.reproducibility.validate_analysis_artifacts import (
    analysis_record_key,
    read_jsonl,
    validate_analysis_records,
    validate_cross_stage_identity,
    validate_decision_records,
)
from study.reproducibility.validate_inference_analysis_chain import (
    validate_inference_analysis_chain,
    validate_inference_records,
)
from study.reproducibility.validate_manifest_analysis_chain import (
    CELEB_DATASET,
    FFPP_DATASET,
    build_manifest_index,
    read_manifest_csv,
    strict_csv_binary_int,
    validate_manifest_analysis_chain,
)
from study.reproducibility.validate_environment_snapshot import (
    validate_environment_snapshot,
)


REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


EXPECTED_DETECTORS = {
    "xception",
    "ucf",
    "spsl",
}

EXPECTED_CONDITIONS = {
    "CLN",
    "RSZ",
    "BLR",
    "H40",
    "PLT",
}

EXPECTED_PROCESSED_CONDITIONS = (
    "RSZ",
    "BLR",
    "H40",
    "PLT",
)

EXPECTED_MANIFEST_ROLE_COUNTS = {
    (
        FFPP_DATASET,
        "fit",
    ): 2880,
    (
        FFPP_DATASET,
        "development",
    ): 720,
    (
        FFPP_DATASET,
        "validation",
    ): 700,
    (
        FFPP_DATASET,
        "test",
    ): 700,
    (
        CELEB_DATASET,
        "external_evaluation",
    ): 518,
}

EXPECTED_MANIFEST_LABEL_COUNTS = {
    (
        FFPP_DATASET,
        "fit",
        0,
    ): 576,
    (
        FFPP_DATASET,
        "fit",
        1,
    ): 2304,

    (
        FFPP_DATASET,
        "development",
        0,
    ): 144,
    (
        FFPP_DATASET,
        "development",
        1,
    ): 576,

    (
        FFPP_DATASET,
        "validation",
        0,
    ): 140,
    (
        FFPP_DATASET,
        "validation",
        1,
    ): 560,

    (
        FFPP_DATASET,
        "test",
        0,
    ): 140,
    (
        FFPP_DATASET,
        "test",
        1,
    ): 560,

    (
        CELEB_DATASET,
        "external_evaluation",
        0,
    ): 178,
    (
        CELEB_DATASET,
        "external_evaluation",
        1,
    ): 340,
}

EXPECTED_MANIFEST_RECORDS = 5518


EXPECTED_PROCESSING_RECORDS = 1358

EXPECTED_PROCESSING_DATASET_COUNTS = {
    FFPP_DATASET: 840,
    CELEB_DATASET: 518,
}

EXPECTED_PROCESSING_ROLE_COUNTS = {
    "{}|{}".format(
        FFPP_DATASET,
        "validation",
    ): 140,

    "{}|{}".format(
        FFPP_DATASET,
        "test",
    ): 700,

    "{}|{}".format(
        CELEB_DATASET,
        "test",
    ): 518,
}

EXPECTED_PROCESSING_SUBGROUP_COUNTS = {
    "{}|{}|{}".format(
        FFPP_DATASET,
        "validation",
        "original",
    ): 140,

    "{}|{}|{}".format(
        FFPP_DATASET,
        "test",
        "original",
    ): 140,

    "{}|{}|{}".format(
        FFPP_DATASET,
        "test",
        "Deepfakes",
    ): 140,

    "{}|{}|{}".format(
        FFPP_DATASET,
        "test",
        "Face2Face",
    ): 140,

    "{}|{}|{}".format(
        FFPP_DATASET,
        "test",
        "FaceSwap",
    ): 140,

    "{}|{}|{}".format(
        FFPP_DATASET,
        "test",
        "NeuralTextures",
    ): 140,

    "{}|{}|{}".format(
        CELEB_DATASET,
        "test",
        "Celeb-real",
    ): 108,

    "{}|{}|{}".format(
        CELEB_DATASET,
        "test",
        "Celeb-synthesis",
    ): 340,

    "{}|{}|{}".format(
        CELEB_DATASET,
        "test",
        "YouTube-real",
    ): 70,
}

EXPECTED_PROCESSING_LABEL_COUNTS = {
    "0": 458,
    "1": 900,
}

EXPECTED_PROCESSING_IMPLEMENTATION = {
    "processing_config.yaml",
    "processing_common.py",
    "generate_processed_video.py",
    "validate_processed_video.py",
    "preflight_source_videos.py",
    "preflight_listed_videos.py",
    "build_processing_manifest.py",
    "estimate_processing_storage.py",
    "run_processing_corpus.py",
    "write_processing_provenance.py",
}

PROCESSING_MANIFEST_FIELDS = (
    "dataset",
    "role",
    "subgroup",
    "study_label",
    "source_label",
    "base_video_id",
    "relative_source_path",
    "absolute_source_path",
    "conditions",
)


EXPECTED_PRODUCTION_BASE_VIDEOS = 1358

EXPECTED_PRODUCTION_RECORDS = (
    EXPECTED_PRODUCTION_BASE_VIDEOS
    * len(
        EXPECTED_CONDITIONS
    )
    * len(
        EXPECTED_DETECTORS
    )
)


TRAINING_STEPS_PER_EPOCH = 4608
TRAINING_COMPLETED_EPOCHS = 10
TRAINING_SEED = 1024

TRAINING_SELECTION_METRIC = (
    "video_auc"
)

TRAINING_SELECTION_RULE = (
    "strictly_greater_video_auc_earliest_tie"
)


CHECKPOINT_PATHS = {
    "xception": (
        REPO_ROOT
        / "training"
        / "weights"
        / "xception_ffpp_fit_best.pth"
    ),

    "ucf": (
        REPO_ROOT
        / "training"
        / "weights"
        / "ucf_ffpp_fit_best.pth"
    ),

    "spsl": (
        REPO_ROOT
        / "training"
        / "weights"
        / "spsl_ffpp_fit_best.pth"
    ),
}


AUDIT_IMPLEMENTATION_FILES = (
    "study/reproducibility/validate_analysis_artifacts.py",
    "study/reproducibility/validate_inference_analysis_chain.py",
    "study/reproducibility/validate_manifest_analysis_chain.py",
    "study/reproducibility/validate_environment_snapshot.py",
    "study/reproducibility/write_environment_snapshot.py",
    "study/reproducibility/run_reproducibility_audit.py",
    "study/evaluation/build_temporal_plan.py",
    "study/evaluation/validate_temporal_plan.py",
    "study/evaluation/build_clean_geometry.py",
    "study/evaluation/validate_clean_geometry.py",
    "study/materialization/plan_training_frames.py",
    "study/materialization/face_alignment.py",
)


ACCEPTED_PROCESSING_QC_STATUSES = {
    "existing_valid",
    "generated_valid",
    "repaired_valid",
}


TEMPORAL_PLAN_FIELDS = (
    "dataset",
    "role",
    "subgroup",
    "study_label",
    "source_label",
    "base_video_id",
    "relative_source_path",
    "absolute_source_path",
    "conditions",
    "decoded_frame_count",
    "target_frame_budget",
    "sampling_method",
    "target_indices",
    "temporal_plan_status",
    "failure_stage",
    "failure_reason",
)

TEMPORAL_FRAME_BUDGET = 32
TEMPORAL_SAMPLING_METHOD = "equal_bin_midpoint_v1"

TEMPORAL_PREFLIGHT_RELATIVE_PATHS = (
    (
        "ffpp",
        "original_preflight.csv",
    ),
    (
        "ffpp",
        "Deepfakes_preflight.csv",
    ),
    (
        "ffpp",
        "Face2Face_preflight.csv",
    ),
    (
        "ffpp",
        "FaceSwap_preflight.csv",
    ),
    (
        "ffpp",
        "NeuralTextures_preflight.csv",
    ),
    (
        "celeb_df_v2",
        "test_preflight.csv",
    ),
)

CLEAN_GEOMETRY_FIELDS = (
    "dataset",
    "role",
    "subgroup",
    "study_label",
    "source_label",
    "base_video_id",
    "relative_source_path",
    "absolute_source_path",
    "temporal_position",
    "source_frame_index",
    "source_frame_sha256",
    "alignment_method",
    "geometry_status",
    "face_count",
    "bbox_json",
    "keypoints_json",
    "affine_matrix_json",
    "aligned_height",
    "aligned_width",
    "aligned_channels",
    "aligned_sha256",
    "failure_stage",
    "failure_reason",
)

CLEAN_GEOMETRY_ALIGNMENT_METHOD = (
    "dfb_dlib_5pt_similarity_v1"
)
CLEAN_GEOMETRY_ALIGNMENT_SCALE = 1.3
CLEAN_GEOMETRY_DETECTOR_UPSAMPLE = 1
CLEAN_GEOMETRY_LANDMARK_INDICES = [
    37,
    44,
    30,
    49,
    55,
]
CLEAN_GEOMETRY_OUTPUT_SIZE = 256
CLEAN_GEOMETRY_PREDICTOR_SHA256 = (
    "8cae4375589dd915d9a0a881101bed1bbb4e9887e35e63b024388f1ca25ff869"
)
CLEAN_GEOMETRY_RECORDS = (
    EXPECTED_PROCESSING_RECORDS
    * TEMPORAL_FRAME_BUDGET
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def read_json(
    path: Path,
) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        value = json.load(
            file
        )

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            "JSON artifact must contain "
            "a top-level object: {}".format(
                path
            )
        )

    return value


def read_csv(
    path: Path,
) -> Tuple[
    Tuple[str, ...],
    List[Dict[str, str]],
]:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        fields = tuple(
            reader.fieldnames
            or ()
        )

        rows = list(
            reader
        )

    return (
        fields,
        rows,
    )


def write_json(
    path: Path,
    value: Dict[str, Any],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            value,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write(
            "\n"
        )

    temporary.replace(
        path
    )


def strict_number(
    value: Any,
    *,
    field_name: str,
) -> float:
    if (
        type(value) is not int
        and type(value) is not float
    ):
        raise ValueError(
            "{} must be a JSON number; "
            "got {!r}.".format(
                field_name,
                value,
            )
        )

    parsed = float(
        value
    )

    if not math.isfinite(
        parsed
    ):
        raise ValueError(
            "{} must be finite.".format(
                field_name
            )
        )

    return parsed


def strict_int(
    value: Any,
    *,
    field_name: str,
) -> int:
    if type(value) is not int:
        raise ValueError(
            "{} must be an exact integer; "
            "got {!r}.".format(
                field_name,
                value,
            )
        )

    return value


def strict_bool(
    value: Any,
    *,
    field_name: str,
) -> bool:
    if type(value) is not bool:
        raise ValueError(
            "{} must be an exact boolean; "
            "got {!r}.".format(
                field_name,
                value,
            )
        )

    return value


def strict_csv_label(
    value: Any,
) -> int:
    if type(value) is not str:
        raise ValueError(
            "Processing study_label must "
            "be a CSV string."
        )

    if value not in (
        "0",
        "1",
    ):
        raise ValueError(
            "Processing study_label must "
            "be exactly '0' or '1'; got {!r}."
            .format(
                value
            )
        )

    return (
        0
        if value == "0"
        else 1
    )


def strict_csv_nonnegative_int(
    value: Any,
    *,
    field_name: str,
) -> int:
    if type(value) is not str:
        raise ValueError(
            "{} must be CSV text; got {!r}.".format(
                field_name,
                value,
            )
        )

    if (
        not value
        or not value.isdigit()
    ):
        raise ValueError(
            "{} must be an exact non-negative integer "
            "string; got {!r}.".format(
                field_name,
                value,
            )
        )

    parsed = int(value)

    if str(parsed) != value:
        raise ValueError(
            "{} must use canonical decimal integer "
            "text; got {!r}.".format(
                field_name,
                value,
            )
        )

    return parsed


def is_sha256_text(
    value: Any,
) -> bool:
    if (
        type(value) is not str
        or len(value) != 64
    ):
        return False

    try:
        int(
            value,
            16,
        )
    except ValueError:
        return False

    return True


def temporal_midpoint_indices(
    *,
    frame_count: int,
    budget: int,
) -> List[int]:
    if frame_count < budget:
        raise ValueError(
            "frame_count must be >= budget."
        )

    return [
        (
            (
                2 * position
                + 1
            )
            * frame_count
        )
        // (
            2
            * budget
        )
        for position in range(
            budget
        )
    ]


def expect_number(
    value: Any,
    expected: float,
    *,
    field_name: str,
):
    parsed = strict_number(
        value,
        field_name=field_name,
    )

    if not math.isclose(
        parsed,
        expected,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "{} mismatch: {} != {}.".format(
                field_name,
                parsed,
                expected,
            )
        )


def artifact_fingerprint(
    path: Path,
) -> Dict[str, Any]:
    path = (
        path
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    return {
        "path": str(
            path
        ),

        "sha256": (
            sha256_file(
                path
            )
        ),

        "size_bytes": (
            path.stat().st_size
        ),
    }


def add_check(
    checks: List[
        Dict[str, Any]
    ],
    name: str,
    details: Dict[
        str,
        Any,
    ] = None,
):
    record = {
        "name": name,
        "status": "passed",
    }

    if details is not None:
        record[
            "details"
        ] = details

    checks.append(
        record
    )


def git_command(
    arguments: List[str],
) -> str:
    completed = subprocess.run(
        [
            "git",
        ]
        + arguments,
        cwd=str(
            REPO_ROOT
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    )

    return completed.stdout.strip()


def current_git_state() -> Dict[str, Any]:
    try:
        commit = git_command(
            [
                "rev-parse",
                "HEAD",
            ]
        )

        branch = git_command(
            [
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            ]
        )

        porcelain = git_command(
            [
                "status",
                "--porcelain",
            ]
        )

        return {
            "status_available": True,
            "commit": commit,
            "branch": branch,
            "dirty": bool(
                porcelain
            ),
        }

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return {
            "status_available": False,
            "commit": None,
            "branch": None,
            "dirty": None,
        }


def current_runtime() -> Dict[str, Any]:
    return {
        "python": (
            sys.version.split()[0]
        ),

        "python_implementation": (
            platform.python_implementation()
        ),

        "numpy": np.__version__,

        "platform": platform.platform(),
    }


def validate_manifest_population(
    rows: Iterable[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    rows = list(
        rows
    )

    if (
        len(
            rows
        )
        != EXPECTED_MANIFEST_RECORDS
    ):
        raise ValueError(
            "Canonical manifest record count "
            "mismatch: {} != {}.".format(
                len(
                    rows
                ),
                EXPECTED_MANIFEST_RECORDS,
            )
        )

    role_counts = Counter()
    label_counts = Counter()

    ffpp_roles_by_source_group = (
        defaultdict(
            set
        )
    )

    for row in rows:
        dataset = row[
            "dataset"
        ]

        role = row[
            "role"
        ]

        label = (
            strict_csv_binary_int(
                row[
                    "study_label"
                ],
                field_name="study_label",
            )
        )

        role_counts[
            (
                dataset,
                role,
            )
        ] += 1

        label_counts[
            (
                dataset,
                role,
                label,
            )
        ] += 1

        if dataset == FFPP_DATASET:
            ffpp_roles_by_source_group[
                row[
                    "source_group_id"
                ]
            ].add(
                role
            )

    if (
        dict(
            role_counts
        )
        != EXPECTED_MANIFEST_ROLE_COUNTS
    ):
        raise ValueError(
            "Canonical manifest role counts "
            "do not match the frozen study design."
        )

    if (
        dict(
            label_counts
        )
        != EXPECTED_MANIFEST_LABEL_COUNTS
    ):
        raise ValueError(
            "Canonical manifest role/label counts "
            "do not match the frozen study design."
        )

    cross_role_groups = {
        source_group_id: sorted(
            roles
        )
        for (
            source_group_id,
            roles,
        ) in (
            ffpp_roles_by_source_group.items()
        )
        if len(
            roles
        ) != 1
    }

    if cross_role_groups:
        examples = list(
            sorted(
                cross_role_groups.items()
            )
        )[:5]

        raise ValueError(
            "FF++ source groups occur in multiple "
            "study roles: {}.".format(
                examples
            )
        )

    return {
        "total_records": len(
            rows
        ),

        "role_counts": {
            "{}|{}".format(
                dataset,
                role,
            ): count
            for (
                dataset,
                role,
            ), count in sorted(
                role_counts.items()
            )
        },

        "role_label_counts": {
            "{}|{}|{}".format(
                dataset,
                role,
                label,
            ): count
            for (
                dataset,
                role,
                label,
            ), count in sorted(
                label_counts.items()
            )
        },

        "ffpp_cross_role_source_groups": 0,
    }


def checkpoint_hashes_from_analysis(
    analysis_records: Iterable[
        Dict[str, Any]
    ],
) -> Dict[str, str]:
    values = defaultdict(
        set
    )

    for row in analysis_records:
        detector = row[
            "detector"
        ]

        checkpoint = row[
            "checkpoint_sha256"
        ]

        if (
            type(checkpoint) is not str
            or len(
                checkpoint
            )
            != 64
        ):
            raise ValueError(
                "Invalid checkpoint SHA-256 for "
                "detector {}.".format(
                    detector
                )
            )

        values[
            detector
        ].add(
            checkpoint
        )

    if (
        set(
            values
        )
        != EXPECTED_DETECTORS
    ):
        raise ValueError(
            "Analysis detector set mismatch: {}."
            .format(
                sorted(
                    values
                )
            )
        )

    output = {}

    for detector in sorted(
        EXPECTED_DETECTORS
    ):
        detector_values = values[
            detector
        ]

        if (
            len(
                detector_values
            )
            != 1
        ):
            raise ValueError(
                "Detector {} uses multiple "
                "checkpoint hashes.".format(
                    detector
                )
            )

        recorded_hash = next(
            iter(
                detector_values
            )
        )

        checkpoint_path = (
            CHECKPOINT_PATHS[
                detector
            ]
        )

        actual_hash = sha256_file(
            checkpoint_path
        )

        if (
            actual_hash
            != recorded_hash
        ):
            raise ValueError(
                "Checkpoint file hash mismatch "
                "for {}: {} != {}.".format(
                    detector,
                    actual_hash,
                    recorded_hash,
                )
            )

        output[
            detector
        ] = actual_hash

    return output


def validate_training_provenance(
    *,
    metadata_paths: List[Path],
    checkpoint_hashes: Dict[
        str,
        str,
    ],
) -> Dict[str, Any]:
    if (
        len(
            metadata_paths
        )
        != len(
            EXPECTED_DETECTORS
        )
    ):
        raise ValueError(
            "Exactly three training metadata "
            "artifacts are required."
        )

    by_detector = {}

    for metadata_path in metadata_paths:
        metadata = read_json(
            metadata_path
        )

        detector = metadata.get(
            "model_name"
        )

        if detector not in EXPECTED_DETECTORS:
            raise ValueError(
                "Unexpected training model_name: {!r}."
                .format(
                    detector
                )
            )

        if detector in by_detector:
            raise ValueError(
                "Duplicate training metadata "
                "for detector {}.".format(
                    detector
                )
            )

        if (
            metadata.get(
                "dev_dataset"
            )
            != FFPP_DATASET
        ):
            raise ValueError(
                "{} checkpoint was not selected "
                "on FaceForensics++ DEV.".format(
                    detector
                )
            )

        if (
            metadata.get(
                "selection_metric"
            )
            != TRAINING_SELECTION_METRIC
        ):
            raise ValueError(
                "{} selection metric mismatch."
                .format(
                    detector
                )
            )

        if (
            metadata.get(
                "selection_rule"
            )
            != TRAINING_SELECTION_RULE
        ):
            raise ValueError(
                "{} selection rule mismatch."
                .format(
                    detector
                )
            )

        seed = strict_int(
            metadata.get(
                "manual_seed"
            ),
            field_name=(
                "{}.manual_seed".format(
                    detector
                )
            ),
        )

        if seed != TRAINING_SEED:
            raise ValueError(
                "{} training seed mismatch."
                .format(
                    detector
                )
            )

        selected_epoch = strict_int(
            metadata.get(
                "completed_epoch"
            ),
            field_name=(
                "{}.completed_epoch".format(
                    detector
                )
            ),
        )

        if (
            selected_epoch < 0
            or selected_epoch
            >= TRAINING_COMPLETED_EPOCHS
        ):
            raise ValueError(
                "{} selected epoch is outside "
                "the fixed training budget.".format(
                    detector
                )
            )

        selected_steps = strict_int(
            metadata.get(
                "optimizer_steps_completed"
            ),
            field_name=(
                "{}.optimizer_steps_completed"
                .format(
                    detector
                )
            ),
        )

        expected_selected_steps = (
            (
                selected_epoch
                + 1
            )
            * TRAINING_STEPS_PER_EPOCH
        )

        if (
            selected_steps
            != expected_selected_steps
        ):
            raise ValueError(
                "{} selected checkpoint optimizer "
                "step mismatch: {} != {}.".format(
                    detector,
                    selected_steps,
                    expected_selected_steps,
                )
            )

        selected_auc = strict_number(
            metadata.get(
                "selection_value"
            ),
            field_name=(
                "{}.selection_value".format(
                    detector
                )
            ),
        )

        if not (
            0.0
            <= selected_auc
            <= 1.0
        ):
            raise ValueError(
                "{} selected DEV AUC is outside "
                "[0, 1].".format(
                    detector
                )
            )

        run_root = (
            metadata_path
            .parent
            .parent
        )

        run_checkpoint_path = (
            metadata_path.parent
            / "best.pth"
        )

        history_path = (
            run_root
            / "dev"
            / "dev_history.jsonl"
        )

        if not run_checkpoint_path.is_file():
            raise FileNotFoundError(
                run_checkpoint_path
            )

        if not history_path.is_file():
            raise FileNotFoundError(
                history_path
            )

        run_checkpoint_hash = sha256_file(
            run_checkpoint_path
        )

        if (
            run_checkpoint_hash
            != checkpoint_hashes[
                detector
            ]
        ):
            raise ValueError(
                "{} training-run checkpoint does "
                "not match final inference weight."
                .format(
                    detector
                )
            )

        history = read_jsonl(
            history_path
        )

        if (
            len(
                history
            )
            != TRAINING_COMPLETED_EPOCHS
        ):
            raise ValueError(
                "{} DEV history must contain "
                "{} completed epochs; got {}."
                .format(
                    detector,
                    TRAINING_COMPLETED_EPOCHS,
                    len(
                        history
                    ),
                )
            )

        observed_epochs = []
        observed_values = []

        running_best = float(
            "-inf"
        )

        for index, row in enumerate(
            history
        ):
            epoch = strict_int(
                row.get(
                    "completed_epoch"
                ),
                field_name=(
                    "{}.history[{}].completed_epoch"
                    .format(
                        detector,
                        index,
                    )
                ),
            )

            if epoch != index:
                raise ValueError(
                    "{} DEV history epoch order "
                    "mismatch at index {}.".format(
                        detector,
                        index,
                    )
                )

            steps = strict_int(
                row.get(
                    "optimizer_steps_completed"
                ),
                field_name=(
                    "{}.history[{}]."
                    "optimizer_steps_completed"
                    .format(
                        detector,
                        index,
                    )
                ),
            )

            expected_steps = (
                (
                    epoch
                    + 1
                )
                * TRAINING_STEPS_PER_EPOCH
            )

            if steps != expected_steps:
                raise ValueError(
                    "{} DEV history optimizer-step "
                    "mismatch for epoch {}.".format(
                        detector,
                        epoch,
                    )
                )

            value = strict_number(
                row.get(
                    "selection_value"
                ),
                field_name=(
                    "{}.history[{}].selection_value"
                    .format(
                        detector,
                        index,
                    )
                ),
            )

            improved = strict_bool(
                row.get(
                    "improved_best"
                ),
                field_name=(
                    "{}.history[{}].improved_best"
                    .format(
                        detector,
                        index,
                    )
                ),
            )

            expected_improved = (
                value
                > running_best
            )

            if (
                improved
                is not expected_improved
            ):
                raise ValueError(
                    "{} DEV history improved_best "
                    "violates strict-greater selection "
                    "at epoch {}.".format(
                        detector,
                        epoch,
                    )
                )

            if expected_improved:
                running_best = value

            observed_epochs.append(
                epoch
            )

            observed_values.append(
                value
            )

        maximum_auc = max(
            observed_values
        )

        first_best_epoch = next(
            epoch
            for (
                epoch,
                value,
            ) in zip(
                observed_epochs,
                observed_values,
            )
            if value == maximum_auc
        )

        if (
            first_best_epoch
            != selected_epoch
        ):
            raise ValueError(
                "{} metadata selected epoch {} "
                "but earliest maximum DEV AUC "
                "occurs at epoch {}.".format(
                    detector,
                    selected_epoch,
                    first_best_epoch,
                )
            )

        if not math.isclose(
            selected_auc,
            maximum_auc,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                "{} metadata selection value does "
                "not equal DEV-history maximum."
                .format(
                    detector
                )
            )

        by_detector[
            detector
        ] = {
            "selected_epoch": (
                selected_epoch
            ),

            "selection_value": (
                selected_auc
            ),

            "optimizer_steps_completed": (
                selected_steps
            ),

            "training_seed": seed,

            "completed_epochs": (
                len(
                    history
                )
            ),

            "metadata": (
                artifact_fingerprint(
                    metadata_path
                )
            ),

            "dev_history": (
                artifact_fingerprint(
                    history_path
                )
            ),

            "run_checkpoint": (
                artifact_fingerprint(
                    run_checkpoint_path
                )
            ),

            "final_checkpoint": (
                artifact_fingerprint(
                    CHECKPOINT_PATHS[
                        detector
                    ]
                )
            ),
        }

    if (
        set(
            by_detector
        )
        != EXPECTED_DETECTORS
    ):
        raise ValueError(
            "Training provenance detector set "
            "is incomplete."
        )

    return by_detector


def processing_manifest_summary(
    rows: List[
        Dict[str, str]
    ],
) -> Dict[str, Any]:
    dataset_counts = Counter(
        row[
            "dataset"
        ]
        for row in rows
    )

    role_counts = Counter(
        (
            row[
                "dataset"
            ],
            row[
                "role"
            ],
        )
        for row in rows
    )

    subgroup_counts = Counter(
        (
            row[
                "dataset"
            ],
            row[
                "role"
            ],
            row[
                "subgroup"
            ],
        )
        for row in rows
    )

    label_counts = Counter(
        row[
            "study_label"
        ]
        for row in rows
    )

    return {
        "records": len(
            rows
        ),

        "dataset_counts": {
            key: value
            for key, value in sorted(
                dataset_counts.items()
            )
        },

        "dataset_role_counts": {
            "{}|{}".format(
                dataset,
                role,
            ): value
            for (
                dataset,
                role,
            ), value in sorted(
                role_counts.items()
            )
        },

        "dataset_role_subgroup_counts": {
            "{}|{}|{}".format(
                dataset,
                role,
                subgroup,
            ): value
            for (
                dataset,
                role,
                subgroup,
            ), value in sorted(
                subgroup_counts.items()
            )
        },

        "study_label_counts": {
            key: value
            for key, value in sorted(
                label_counts.items()
            )
        },
    }


def resolve_recorded_path(
    value: Any,
) -> Path:
    if (
        type(value) is not str
        or not value
    ):
        raise ValueError(
            "Recorded provenance path must "
            "be a non-empty string."
        )

    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():
        path = (
            REPO_ROOT
            / path
        )

    return path.resolve()


def validate_recorded_file(
    entry: Any,
    *,
    context: str,
) -> Dict[str, Any]:
    if not isinstance(
        entry,
        dict,
    ):
        raise ValueError(
            "{} must be a file-record object."
            .format(
                context
            )
        )

    path = resolve_recorded_path(
        entry.get(
            "path"
        )
    )

    fingerprint = (
        artifact_fingerprint(
            path
        )
    )

    if (
        entry.get(
            "sha256"
        )
        != fingerprint[
            "sha256"
        ]
    ):
        raise ValueError(
            "{} SHA-256 mismatch.".format(
                context
            )
        )

    if (
        entry.get(
            "size_bytes"
        )
        != fingerprint[
            "size_bytes"
        ]
    ):
        raise ValueError(
            "{} size mismatch.".format(
                context
            )
        )

    return fingerprint


def validate_processing_provenance(
    *,
    provenance_path: Path,
    processing_manifest_path: Path,
    canonical_manifest_rows: List[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    provenance = read_json(
        provenance_path
    )

    if (
        provenance.get(
            "schema_version"
        )
        != 1
    ):
        raise ValueError(
            "Unexpected processing provenance "
            "schema version."
        )

    (
        fields,
        rows,
    ) = read_csv(
        processing_manifest_path
    )

    if fields != PROCESSING_MANIFEST_FIELDS:
        raise ValueError(
            "Processing manifest schema mismatch.\n"
            "Expected: {}\n"
            "Actual:   {}".format(
                PROCESSING_MANIFEST_FIELDS,
                fields,
            )
        )

    if (
        len(
            rows
        )
        != EXPECTED_PROCESSING_RECORDS
    ):
        raise ValueError(
            "Processing manifest record count "
            "mismatch."
        )

    processing_index = {}

    for row_number, row in enumerate(
        rows,
        start=1,
    ):
        label = strict_csv_label(
            row.get(
                "study_label"
            )
        )

        conditions = row.get(
            "conditions"
        )

        if conditions != ";".join(
            EXPECTED_PROCESSED_CONDITIONS
        ):
            raise ValueError(
                "Processing manifest row {} has "
                "unexpected condition set: {!r}."
                .format(
                    row_number,
                    conditions,
                )
            )

        key = (
            row[
                "dataset"
            ],
            row[
                "relative_source_path"
            ],
        )

        if key in processing_index:
            raise ValueError(
                "Duplicate processing-manifest "
                "identity: {}.".format(
                    key
                )
            )

        row[
            "_parsed_study_label"
        ] = label

        processing_index[
            key
        ] = row

    actual_summary = (
        processing_manifest_summary(
            rows
        )
    )

    expected_summary = {
        "records": (
            EXPECTED_PROCESSING_RECORDS
        ),

        "dataset_counts": (
            EXPECTED_PROCESSING_DATASET_COUNTS
        ),

        "dataset_role_counts": (
            EXPECTED_PROCESSING_ROLE_COUNTS
        ),

        "dataset_role_subgroup_counts": (
            EXPECTED_PROCESSING_SUBGROUP_COUNTS
        ),

        "study_label_counts": (
            EXPECTED_PROCESSING_LABEL_COUNTS
        ),
    }

    if (
        actual_summary
        != expected_summary
    ):
        raise ValueError(
            "Processing manifest population "
            "does not match the frozen study design."
        )

    #
    # Build the exact canonical population which is
    # allowed to enter processing:
    #
    #   * unique FF++ validation real videos;
    #   * all FF++ test videos;
    #   * all Celeb-DF-v2 external-evaluation videos.
    #
    expected_canonical = {}

    for row in canonical_manifest_rows:
        dataset = row[
            "dataset"
        ]

        role = row[
            "role"
        ]

        label = (
            strict_csv_binary_int(
                row[
                    "study_label"
                ],
                field_name="study_label",
            )
        )

        selected = False

        if dataset == FFPP_DATASET:
            if role == "test":
                selected = True

            elif (
                role == "validation"
                and label == 0
            ):
                selected = True

        elif (
            dataset == CELEB_DATASET
            and role
            == "external_evaluation"
        ):
            selected = True

        if not selected:
            continue

        key = (
            dataset,
            row[
                "relative_source_path"
            ],
        )

        expected_canonical[
            key
        ] = row

    if (
        len(
            expected_canonical
        )
        != EXPECTED_PROCESSING_RECORDS
    ):
        raise ValueError(
            "Canonical processing universe "
            "does not contain 1358 videos."
        )

    if (
        set(
            processing_index
        )
        != set(
            expected_canonical
        )
    ):
        missing = sorted(
            set(
                expected_canonical
            )
            - set(
                processing_index
            )
        )

        unexpected = sorted(
            set(
                processing_index
            )
            - set(
                expected_canonical
            )
        )

        raise ValueError(
            "Processing/canonical universe mismatch; "
            "missing={} unexpected={}."
            .format(
                missing[:5],
                unexpected[:5],
            )
        )

    for key in sorted(
        expected_canonical
    ):
        canonical = (
            expected_canonical[
                key
            ]
        )

        processing = (
            processing_index[
                key
            ]
        )

        canonical_label = (
            strict_csv_binary_int(
                canonical[
                    "study_label"
                ],
                field_name="study_label",
            )
        )

        if (
            processing[
                "_parsed_study_label"
            ]
            != canonical_label
        ):
            raise ValueError(
                "Processing study_label differs "
                "from canonical manifest for {}."
                .format(
                    key
                )
            )

        if (
            processing[
                "base_video_id"
            ]
            != canonical[
                "base_video_id"
            ]
        ):
            raise ValueError(
                "Processing base_video_id differs "
                "from canonical manifest for {}."
                .format(
                    key
                )
            )

        expected_processing_role = (
            "test"
            if canonical[
                "dataset"
            ] == CELEB_DATASET
            else canonical[
                "role"
            ]
        )

        if (
            processing[
                "role"
            ]
            != expected_processing_role
        ):
            raise ValueError(
                "Processing role mismatch "
                "for {}.".format(
                    key
                )
            )

    frozen_inputs = provenance.get(
        "frozen_inputs"
    )

    if not isinstance(
        frozen_inputs,
        dict,
    ):
        raise ValueError(
            "Processing provenance has no "
            "frozen_inputs."
        )

    processing_manifest_record = (
        frozen_inputs.get(
            "processing_manifest"
        )
    )

    recorded_processing_manifest = (
        validate_recorded_file(
            processing_manifest_record,
            context=(
                "processing_provenance."
                "processing_manifest"
            ),
        )
    )

    actual_processing_manifest = (
        artifact_fingerprint(
            processing_manifest_path
        )
    )

    if (
        recorded_processing_manifest[
            "sha256"
        ]
        != actual_processing_manifest[
            "sha256"
        ]
    ):
        raise ValueError(
            "Processing provenance references "
            "a different processing manifest."
        )

    ffpp_splits = frozen_inputs.get(
        "ffpp_official_splits"
    )

    if not isinstance(
        ffpp_splits,
        dict,
    ):
        raise ValueError(
            "Processing provenance has no "
            "FF++ official split fingerprints."
        )

    if (
        set(
            ffpp_splits
        )
        != {
            "train.json",
            "val.json",
            "test.json",
        }
    ):
        raise ValueError(
            "Processing provenance FF++ split "
            "set mismatch."
        )

    split_fingerprints = {}

    for name in sorted(
        ffpp_splits
    ):
        split_fingerprints[
            name
        ] = validate_recorded_file(
            ffpp_splits[
                name
            ],
            context=(
                "processing_provenance."
                "ffpp_official_splits."
                + name
            ),
        )

    celeb_test_list = (
        validate_recorded_file(
            frozen_inputs.get(
                "celeb_df_v2_official_test_list"
            ),
            context=(
                "processing_provenance."
                "celeb_df_v2_official_test_list"
            ),
        )
    )

    implementation = provenance.get(
        "processing_implementation"
    )

    if not isinstance(
        implementation,
        dict,
    ):
        raise ValueError(
            "Processing provenance has no "
            "implementation fingerprints."
        )

    if (
        set(
            implementation
        )
        != EXPECTED_PROCESSING_IMPLEMENTATION
    ):
        missing = sorted(
            EXPECTED_PROCESSING_IMPLEMENTATION
            - set(
                implementation
            )
        )

        unexpected = sorted(
            set(
                implementation
            )
            - EXPECTED_PROCESSING_IMPLEMENTATION
        )

        raise ValueError(
            "Processing implementation fingerprint "
            "set mismatch; missing={} unexpected={}."
            .format(
                missing,
                unexpected,
            )
        )

    implementation_fingerprints = {}

    for name in sorted(
        implementation
    ):
        implementation_fingerprints[
            name
        ] = validate_recorded_file(
            implementation[
                name
            ],
            context=(
                "processing_provenance."
                "processing_implementation."
                + name
            ),
        )

    recorded_summary = provenance.get(
        "manifest_summary"
    )

    if (
        recorded_summary
        != actual_summary
    ):
        raise ValueError(
            "Processing provenance manifest_summary "
            "does not match the current manifest."
        )

    runtime = provenance.get(
        "runtime"
    )

    if not isinstance(
        runtime,
        dict,
    ):
        raise ValueError(
            "Processing provenance has no "
            "runtime metadata."
        )

    for field_name in (
        "python",
        "python_executable",
        "platform",
        "opencv",
        "numpy",
        "pyyaml",
        "ffmpeg",
        "ffprobe",
    ):
        value = runtime.get(
            field_name
        )

        if (
            type(value) is not str
            or not value
        ):
            raise ValueError(
                "Processing runtime metadata "
                "is missing {}.".format(
                    field_name
                )
            )

    repository = provenance.get(
        "repository"
    )

    if not isinstance(
        repository,
        dict,
    ):
        raise ValueError(
            "Processing provenance has no "
            "repository metadata."
        )

    for field_name in (
        "root",
        "commit",
        "branch",
    ):
        if not repository.get(
            field_name
        ):
            raise ValueError(
                "Processing repository metadata "
                "is missing {}.".format(
                    field_name
                )
            )

    strict_bool(
        repository.get(
            "working_tree_dirty"
        ),
        field_name=(
            "processing.repository."
            "working_tree_dirty"
        ),
    )

    return {
        "processing_manifest": (
            actual_processing_manifest
        ),

        "population": (
            actual_summary
        ),

        "canonical_universe_match": True,

        "official_ffpp_splits": (
            split_fingerprints
        ),

        "celeb_df_v2_test_list": (
            celeb_test_list
        ),

        "implementation_file_count": (
            len(
                implementation_fingerprints
            )
        ),

        "runtime": runtime,

        "repository": repository,
    }


def validate_processing_qc(
    *,
    summary_paths: List[Path],
    mode: str,
) -> Dict[str, Any]:
    if (
        len(
            summary_paths
        )
        != len(
            EXPECTED_PROCESSED_CONDITIONS
        )
    ):
        raise ValueError(
            "Exactly four processing QC summaries "
            "are required."
        )

    output = {}

    expected_selected = (
        4
        if mode == "smoke"
        else EXPECTED_PROCESSING_RECORDS
    )

    for summary_path in summary_paths:
        summary = read_json(
            summary_path
        )

        condition = summary.get(
            "condition"
        )

        if (
            condition
            not in EXPECTED_PROCESSED_CONDITIONS
        ):
            raise ValueError(
                "Unexpected processing QC "
                "condition: {!r}.".format(
                    condition
                )
            )

        if condition in output:
            raise ValueError(
                "Duplicate processing QC summary "
                "for condition {}.".format(
                    condition
                )
            )

        selected_videos = strict_int(
            summary.get(
                "selected_videos"
            ),
            field_name=(
                "{}.selected_videos".format(
                    condition
                )
            ),
        )

        if (
            selected_videos
            != expected_selected
        ):
            raise ValueError(
                "{} QC selected-video count "
                "mismatch: {} != {}.".format(
                    condition,
                    selected_videos,
                    expected_selected,
                )
            )

        status_counts = summary.get(
            "status_counts"
        )

        if not isinstance(
            status_counts,
            dict,
        ):
            raise ValueError(
                "{} QC summary has no "
                "status_counts.".format(
                    condition
                )
            )

        if not status_counts:
            raise ValueError(
                "{} QC summary has empty "
                "status_counts.".format(
                    condition
                )
            )

        total_status = 0

        for (
            status,
            count,
        ) in status_counts.items():
            if (
                status
                not in ACCEPTED_PROCESSING_QC_STATUSES
            ):
                raise ValueError(
                    "{} QC contains non-valid "
                    "status {!r}.".format(
                        condition,
                        status,
                    )
                )

            count = strict_int(
                count,
                field_name=(
                    "{}.status_counts.{}"
                    .format(
                        condition,
                        status,
                    )
                ),
            )

            if count < 0:
                raise ValueError(
                    "Negative processing QC count."
                )

            total_status += count

        if (
            total_status
            != selected_videos
        ):
            raise ValueError(
                "{} QC status total mismatch."
                .format(
                    condition
                )
            )

        qc_jsonl_value = summary.get(
            "qc_jsonl"
        )

        if (
            type(qc_jsonl_value) is not str
            or not qc_jsonl_value
        ):
            raise ValueError(
                "{} QC summary has invalid "
                "qc_jsonl path.".format(
                    condition
                )
            )

        qc_jsonl_path = (
            Path(
                qc_jsonl_value
            )
            .expanduser()
            .resolve()
        )

        qc_records = read_jsonl(
            qc_jsonl_path
        )

        if (
            len(
                qc_records
            )
            != selected_videos
        ):
            raise ValueError(
                "{} QC JSONL record count "
                "mismatch.".format(
                    condition
                )
            )

        observed_status_counts = Counter(
            row.get(
                "status"
            )
            for row in qc_records
        )

        if (
            dict(
                observed_status_counts
            )
            != status_counts
        ):
            raise ValueError(
                "{} QC JSONL statuses differ "
                "from summary.".format(
                    condition
                )
            )

        output[
            condition
        ] = {
            "selected_videos": (
                selected_videos
            ),

            "status_counts": (
                status_counts
            ),

            "summary": (
                artifact_fingerprint(
                    summary_path
                )
            ),

            "qc_jsonl": (
                artifact_fingerprint(
                    qc_jsonl_path
                )
            ),
        }

    if (
        set(
            output
        )
        != set(
            EXPECTED_PROCESSED_CONDITIONS
        )
    ):
        raise ValueError(
            "Processing QC condition set "
            "is incomplete."
        )

    return {
        condition: output[
            condition
        ]
        for condition in (
            EXPECTED_PROCESSED_CONDITIONS
        )
    }


def validate_temporal_plan_artifacts(
    *,
    plan_path: Path,
    summary_path: Path,
    processing_manifest_path: Path,
) -> Dict[str, Any]:
    (
        fields,
        plan_rows,
    ) = read_csv(
        plan_path
    )

    if fields != TEMPORAL_PLAN_FIELDS:
        raise ValueError(
            "Temporal-plan schema mismatch.\n"
            "Expected: {}\n"
            "Actual:   {}".format(
                TEMPORAL_PLAN_FIELDS,
                fields,
            )
        )

    if (
        len(
            plan_rows
        )
        != EXPECTED_PROCESSING_RECORDS
    ):
        raise ValueError(
            "Temporal-plan record count mismatch: "
            "{} != {}.".format(
                len(
                    plan_rows
                ),
                EXPECTED_PROCESSING_RECORDS,
            )
        )

    (
        _,
        processing_rows,
    ) = read_csv(
        processing_manifest_path
    )

    processing_index = {}

    for row in processing_rows:
        key = (
            row[
                "dataset"
            ],
            row[
                "relative_source_path"
            ],
        )

        if key in processing_index:
            raise ValueError(
                "Duplicate processing-manifest "
                "identity while validating temporal "
                "plan: {}.".format(
                    key
                )
            )

        processing_index[
            key
        ] = row

    plan_index = {}
    decoded_counts = []

    for row_number, row in enumerate(
        plan_rows,
        start=1,
    ):
        key = (
            row[
                "dataset"
            ],
            row[
                "relative_source_path"
            ],
        )

        if key in plan_index:
            raise ValueError(
                "Duplicate temporal-plan identity: "
                "{}.".format(
                    key
                )
            )

        manifest_row = (
            processing_index.get(
                key
            )
        )

        if manifest_row is None:
            raise ValueError(
                "Temporal-plan row is absent from "
                "processing manifest: {}.".format(
                    key
                )
            )

        for field_name in (
            "dataset",
            "role",
            "subgroup",
            "study_label",
            "source_label",
            "base_video_id",
            "relative_source_path",
            "absolute_source_path",
            "conditions",
        ):
            if (
                row[
                    field_name
                ]
                != manifest_row[
                    field_name
                ]
            ):
                raise ValueError(
                    "Temporal-plan/processing-manifest "
                    "mismatch for {} field {}.".format(
                        key,
                        field_name,
                    )
                )

        strict_csv_label(
            row[
                "study_label"
            ]
        )

        decoded_count = (
            strict_csv_nonnegative_int(
                row[
                    "decoded_frame_count"
                ],
                field_name=(
                    "temporal_plan[{}]."
                    "decoded_frame_count".format(
                        row_number
                    )
                ),
            )
        )

        if (
            decoded_count
            < TEMPORAL_FRAME_BUDGET
        ):
            raise ValueError(
                "Temporal-plan decoded frame count "
                "is below 32 for {}.".format(
                    key
                )
            )

        target_budget = (
            strict_csv_nonnegative_int(
                row[
                    "target_frame_budget"
                ],
                field_name=(
                    "temporal_plan[{}]."
                    "target_frame_budget".format(
                        row_number
                    )
                ),
            )
        )

        if (
            target_budget
            != TEMPORAL_FRAME_BUDGET
        ):
            raise ValueError(
                "Temporal-plan target-frame budget "
                "mismatch for {}.".format(
                    key
                )
            )

        if (
            row[
                "sampling_method"
            ]
            != TEMPORAL_SAMPLING_METHOD
        ):
            raise ValueError(
                "Temporal sampling method mismatch "
                "for {}.".format(
                    key
                )
            )

        if (
            row[
                "temporal_plan_status"
            ]
            != "ok"
        ):
            raise ValueError(
                "Frozen temporal plan contains "
                "non-ok record for {}.".format(
                    key
                )
            )

        if (
            row[
                "failure_stage"
            ].strip()
            or row[
                "failure_reason"
            ].strip()
        ):
            raise ValueError(
                "Successful temporal-plan row contains "
                "failure metadata for {}.".format(
                    key
                )
            )

        try:
            target_indices = json.loads(
                row[
                    "target_indices"
                ]
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid target_indices JSON for {}."
                .format(
                    key
                )
            ) from exc

        if (
            not isinstance(
                target_indices,
                list,
            )
            or len(
                target_indices
            )
            != TEMPORAL_FRAME_BUDGET
        ):
            raise ValueError(
                "Temporal target-index count mismatch "
                "for {}.".format(
                    key
                )
            )

        if not all(
            type(value) is int
            for value in target_indices
        ):
            raise ValueError(
                "Temporal indices must be exact JSON "
                "integers for {}.".format(
                    key
                )
            )

        expected_indices = (
            temporal_midpoint_indices(
                frame_count=(
                    decoded_count
                ),
                budget=(
                    TEMPORAL_FRAME_BUDGET
                ),
            )
        )

        if (
            target_indices
            != expected_indices
        ):
            raise ValueError(
                "Temporal midpoint positions do not "
                "match the frozen sampling rule for "
                "{}.".format(
                    key
                )
            )

        plan_index[
            key
        ] = {
            "row": row,
            "target_indices": (
                target_indices
            ),
        }

        decoded_counts.append(
            decoded_count
        )

    if (
        set(
            plan_index
        )
        != set(
            processing_index
        )
    ):
        raise ValueError(
            "Temporal-plan membership differs from "
            "processing-manifest membership."
        )

    summary = read_json(
        summary_path
    )

    if (
        summary.get(
            "schema_version"
        )
        != 1
    ):
        raise ValueError(
            "Unexpected temporal-plan summary "
            "schema version."
        )

    expected_dataset_role_counts = (
        EXPECTED_PROCESSING_ROLE_COUNTS
    )

    expected_subgroup_counts = (
        EXPECTED_PROCESSING_SUBGROUP_COUNTS
    )

    expected_summary_values = {
        "base_video_count": (
            EXPECTED_PROCESSING_RECORDS
        ),
        "target_frame_budget": (
            TEMPORAL_FRAME_BUDGET
        ),
        "sampling_method": (
            TEMPORAL_SAMPLING_METHOD
        ),
        "status_counts": {
            "ok": (
                EXPECTED_PROCESSING_RECORDS
            ),
        },
        "minimum_decoded_frame_count": (
            min(
                decoded_counts
            )
        ),
        "maximum_decoded_frame_count": (
            max(
                decoded_counts
            )
        ),
        "dataset_role_counts": (
            expected_dataset_role_counts
        ),
        "subgroup_counts": (
            expected_subgroup_counts
        ),
    }

    for (
        field_name,
        expected_value,
    ) in expected_summary_values.items():
        if (
            summary.get(
                field_name
            )
            != expected_value
        ):
            raise ValueError(
                "Temporal-plan summary mismatch "
                "for {}.".format(
                    field_name
                )
            )

    input_hashes = summary.get(
        "input_hashes"
    )

    if not isinstance(
        input_hashes,
        dict,
    ):
        raise ValueError(
            "Temporal-plan summary has no "
            "input_hashes."
        )

    if (
        input_hashes.get(
            "processing_manifest.csv"
        )
        != sha256_file(
            processing_manifest_path
        )
    ):
        raise ValueError(
            "Temporal-plan processing-manifest "
            "fingerprint mismatch."
        )

    preflight_hashes = input_hashes.get(
        "preflight"
    )

    if not isinstance(
        preflight_hashes,
        dict,
    ):
        raise ValueError(
            "Temporal-plan summary has no "
            "preflight fingerprints."
        )

    study_root = (
        plan_path
        .parent
        .parent
    )

    expected_preflight_names = {
        filename
        for (
            _,
            filename,
        ) in (
            TEMPORAL_PREFLIGHT_RELATIVE_PATHS
        )
    }

    if (
        set(
            preflight_hashes
        )
        != expected_preflight_names
    ):
        raise ValueError(
            "Temporal-plan preflight fingerprint "
            "set mismatch."
        )

    verified_preflight = {}

    for (
        directory,
        filename,
    ) in TEMPORAL_PREFLIGHT_RELATIVE_PATHS:
        path = (
            study_root
            / "preflight"
            / directory
            / filename
        )

        fingerprint = (
            artifact_fingerprint(
                path
            )
        )

        if (
            preflight_hashes[
                filename
            ]
            != fingerprint[
                "sha256"
            ]
        ):
            raise ValueError(
                "Temporal-plan preflight SHA-256 "
                "mismatch for {}.".format(
                    filename
                )
            )

        verified_preflight[
            filename
        ] = fingerprint

    plan_fingerprint = (
        artifact_fingerprint(
            plan_path
        )
    )

    if (
        summary.get(
            "temporal_plan_sha256"
        )
        != plan_fingerprint[
            "sha256"
        ]
    ):
        raise ValueError(
            "Temporal-plan SHA-256 mismatch."
        )

    return {
        "plan": plan_fingerprint,
        "summary": (
            artifact_fingerprint(
                summary_path
            )
        ),
        "base_video_count": (
            len(
                plan_rows
            )
        ),
        "target_frame_budget": (
            TEMPORAL_FRAME_BUDGET
        ),
        "sampling_method": (
            TEMPORAL_SAMPLING_METHOD
        ),
        "minimum_decoded_frame_count": (
            min(
                decoded_counts
            )
        ),
        "maximum_decoded_frame_count": (
            max(
                decoded_counts
            )
        ),
        "preflight": verified_preflight,
        "index": plan_index,
    }


def validate_clean_geometry_artifacts(
    *,
    geometry_path: Path,
    summary_path: Path,
    temporal_plan_path: Path,
    temporal_plan_details: Dict[
        str,
        Any,
    ],
) -> Dict[str, Any]:
    (
        fields,
        geometry_rows,
    ) = read_csv(
        geometry_path
    )

    if fields != CLEAN_GEOMETRY_FIELDS:
        raise ValueError(
            "Clean-geometry schema mismatch.\n"
            "Expected: {}\n"
            "Actual:   {}".format(
                CLEAN_GEOMETRY_FIELDS,
                fields,
            )
        )

    if (
        len(
            geometry_rows
        )
        != CLEAN_GEOMETRY_RECORDS
    ):
        raise ValueError(
            "Clean-geometry record count mismatch: "
            "{} != {}.".format(
                len(
                    geometry_rows
                ),
                CLEAN_GEOMETRY_RECORDS,
            )
        )

    temporal_index = (
        temporal_plan_details[
            "index"
        ]
    )

    geometry_positions = defaultdict(
        set
    )
    valid_counts = Counter()
    status_counts = Counter()
    failure_stages = Counter()

    for row_number, row in enumerate(
        geometry_rows,
        start=1,
    ):
        key = (
            row[
                "dataset"
            ],
            row[
                "relative_source_path"
            ],
        )

        temporal_entry = (
            temporal_index.get(
                key
            )
        )

        if temporal_entry is None:
            raise ValueError(
                "Clean-geometry row is absent from "
                "temporal plan: {}.".format(
                    key
                )
            )

        temporal_row = temporal_entry[
            "row"
        ]

        for field_name in (
            "dataset",
            "role",
            "subgroup",
            "study_label",
            "source_label",
            "base_video_id",
            "relative_source_path",
            "absolute_source_path",
        ):
            if (
                row[
                    field_name
                ]
                != temporal_row[
                    field_name
                ]
            ):
                raise ValueError(
                    "Clean-geometry/temporal-plan "
                    "mismatch for {} field {}.".format(
                        key,
                        field_name,
                    )
                )

        strict_csv_label(
            row[
                "study_label"
            ]
        )

        position = (
            strict_csv_nonnegative_int(
                row[
                    "temporal_position"
                ],
                field_name=(
                    "clean_geometry[{}]."
                    "temporal_position".format(
                        row_number
                    )
                ),
            )
        )

        if (
            position
            >= TEMPORAL_FRAME_BUDGET
        ):
            raise ValueError(
                "Clean-geometry temporal position "
                "is outside 0..31 for {}.".format(
                    key
                )
            )

        if (
            position
            in geometry_positions[
                key
            ]
        ):
            raise ValueError(
                "Duplicate clean-geometry position "
                "for {} position {}.".format(
                    key,
                    position,
                )
            )

        geometry_positions[
            key
        ].add(
            position
        )

        source_frame_index = (
            strict_csv_nonnegative_int(
                row[
                    "source_frame_index"
                ],
                field_name=(
                    "clean_geometry[{}]."
                    "source_frame_index".format(
                        row_number
                    )
                ),
            )
        )

        expected_frame_index = (
            temporal_entry[
                "target_indices"
            ][
                position
            ]
        )

        if (
            source_frame_index
            != expected_frame_index
        ):
            raise ValueError(
                "Clean-geometry source-frame index "
                "differs from temporal plan for {} "
                "position {}.".format(
                    key,
                    position,
                )
            )

        if not is_sha256_text(
            row[
                "source_frame_sha256"
            ]
        ):
            raise ValueError(
                "Invalid clean source-frame SHA-256 "
                "for {} position {}.".format(
                    key,
                    position,
                )
            )

        if (
            row[
                "alignment_method"
            ]
            != CLEAN_GEOMETRY_ALIGNMENT_METHOD
        ):
            raise ValueError(
                "Clean-geometry alignment method "
                "mismatch for {} position {}.".format(
                    key,
                    position,
                )
            )

        status = row[
            "geometry_status"
        ]

        if status not in (
            "valid",
            "invalid",
        ):
            raise ValueError(
                "Unknown geometry_status {!r} for "
                "{} position {}.".format(
                    status,
                    key,
                    position,
                )
            )

        status_counts[
            status
        ] += 1

        if status == "invalid":
            if not row[
                "failure_stage"
            ].strip():
                raise ValueError(
                    "Invalid clean geometry has no "
                    "failure_stage for {} position {}."
                    .format(
                        key,
                        position,
                    )
                )

            if not row[
                "failure_reason"
            ].strip():
                raise ValueError(
                    "Invalid clean geometry has no "
                    "failure_reason for {} position {}."
                    .format(
                        key,
                        position,
                    )
                )

            if any(
                row[
                    field_name
                ].strip()
                for field_name in (
                    "aligned_height",
                    "aligned_width",
                    "aligned_channels",
                    "aligned_sha256",
                )
            ):
                raise ValueError(
                    "Invalid clean geometry contains "
                    "aligned-output metadata for {} "
                    "position {}.".format(
                        key,
                        position,
                    )
                )

            failure_stages[
                row[
                    "failure_stage"
                ]
            ] += 1

            continue

        valid_counts[
            key
        ] += 1

        if (
            row[
                "failure_stage"
            ].strip()
            or row[
                "failure_reason"
            ].strip()
        ):
            raise ValueError(
                "Valid clean geometry contains "
                "failure metadata for {} position {}."
                .format(
                    key,
                    position,
                )
            )

        face_count = (
            strict_csv_nonnegative_int(
                row[
                    "face_count"
                ],
                field_name=(
                    "clean_geometry[{}].face_count"
                    .format(
                        row_number
                    )
                ),
            )
        )

        if face_count < 1:
            raise ValueError(
                "Valid clean geometry has face_count "
                "< 1 for {} position {}.".format(
                    key,
                    position,
                )
            )

        try:
            bbox = json.loads(
                row[
                    "bbox_json"
                ]
            )
            keypoints = np.asarray(
                json.loads(
                    row[
                        "keypoints_json"
                    ]
                ),
                dtype=np.float64,
            )
            affine = np.asarray(
                json.loads(
                    row[
                        "affine_matrix_json"
                    ]
                ),
                dtype=np.float64,
            )
        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "Invalid clean-geometry JSON metadata "
                "for {} position {}.".format(
                    key,
                    position,
                )
            ) from exc

        if (
            not isinstance(
                bbox,
                list,
            )
            or len(
                bbox
            )
            != 4
            or not all(
                type(value) is int
                for value in bbox
            )
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
        ):
            raise ValueError(
                "Invalid clean-geometry bbox for {} "
                "position {}.".format(
                    key,
                    position,
                )
            )

        if (
            keypoints.shape
            != (
                5,
                2,
            )
            or not np.isfinite(
                keypoints
            ).all()
        ):
            raise ValueError(
                "Invalid clean-geometry keypoints for "
                "{} position {}.".format(
                    key,
                    position,
                )
            )

        if (
            affine.shape
            != (
                2,
                3,
            )
            or not np.isfinite(
                affine
            ).all()
        ):
            raise ValueError(
                "Invalid clean-geometry affine matrix "
                "for {} position {}.".format(
                    key,
                    position,
                )
            )

        dimensions = (
            strict_csv_nonnegative_int(
                row[
                    "aligned_height"
                ],
                field_name="aligned_height",
            ),
            strict_csv_nonnegative_int(
                row[
                    "aligned_width"
                ],
                field_name="aligned_width",
            ),
            strict_csv_nonnegative_int(
                row[
                    "aligned_channels"
                ],
                field_name="aligned_channels",
            ),
        )

        if dimensions != (
            CLEAN_GEOMETRY_OUTPUT_SIZE,
            CLEAN_GEOMETRY_OUTPUT_SIZE,
            3,
        ):
            raise ValueError(
                "Unexpected aligned clean-geometry "
                "shape for {} position {}: {}.".format(
                    key,
                    position,
                    dimensions,
                )
            )

        if not is_sha256_text(
            row[
                "aligned_sha256"
            ]
        ):
            raise ValueError(
                "Invalid aligned clean-geometry "
                "SHA-256 for {} position {}.".format(
                    key,
                    position,
                )
            )

    if (
        set(
            geometry_positions
        )
        != set(
            temporal_index
        )
    ):
        raise ValueError(
            "Clean-geometry video membership differs "
            "from temporal-plan membership."
        )

    expected_positions = set(
        range(
            TEMPORAL_FRAME_BUDGET
        )
    )

    for key in temporal_index:
        if (
            geometry_positions[
                key
            ]
            != expected_positions
        ):
            raise ValueError(
                "Clean geometry does not contain all "
                "32 temporal positions for {}.".format(
                    key
                )
            )

    valid_counts_by_video = {
        key: valid_counts.get(
            key,
            0,
        )
        for key in temporal_index
    }

    distribution = Counter(
        valid_counts_by_video.values()
    )

    distribution_json = {
        str(
            valid_count
        ): video_count
        for (
            valid_count,
            video_count,
        ) in sorted(
            distribution.items()
        )
    }

    geometry_fingerprint = (
        artifact_fingerprint(
            geometry_path
        )
    )

    temporal_fingerprint = (
        artifact_fingerprint(
            temporal_plan_path
        )
    )

    predictor_path = (
        REPO_ROOT
        / "preprocessing"
        / "dlib_tools"
        / "shape_predictor_81_face_landmarks.dat"
    )

    predictor_fingerprint = (
        artifact_fingerprint(
            predictor_path
        )
    )

    if (
        predictor_fingerprint[
            "sha256"
        ]
        != CLEAN_GEOMETRY_PREDICTOR_SHA256
    ):
        raise ValueError(
            "Frozen dlib predictor SHA-256 mismatch."
        )

    summary = read_json(
        summary_path
    )

    if (
        summary.get(
            "schema_version"
        )
        != 1
    ):
        raise ValueError(
            "Unexpected clean-geometry summary "
            "schema version."
        )

    expected_exact = {
        "alignment_method": (
            CLEAN_GEOMETRY_ALIGNMENT_METHOD
        ),
        "detector_upsample": (
            CLEAN_GEOMETRY_DETECTOR_UPSAMPLE
        ),
        "landmark_indices": (
            CLEAN_GEOMETRY_LANDMARK_INDICES
        ),
        "output_size": (
            CLEAN_GEOMETRY_OUTPUT_SIZE
        ),
        "predictor_sha256": (
            CLEAN_GEOMETRY_PREDICTOR_SHA256
        ),
        "selected_video_count": (
            EXPECTED_PROCESSING_RECORDS
        ),
        "targets_per_video": (
            TEMPORAL_FRAME_BUDGET
        ),
        "nominal_target_count": (
            CLEAN_GEOMETRY_RECORDS
        ),
        "geometry_status_counts": dict(
            sorted(
                status_counts.items()
            )
        ),
        "failure_stages": dict(
            sorted(
                failure_stages.items()
            )
        ),
        "clean_valid_frame_count_distribution": (
            distribution_json
        ),
        "minimum_clean_valid_frames": (
            min(
                valid_counts_by_video.values()
            )
        ),
        "maximum_clean_valid_frames": (
            max(
                valid_counts_by_video.values()
            )
        ),
        "complete_32_video_count": (
            distribution.get(
                TEMPORAL_FRAME_BUDGET,
                0,
            )
        ),
        "zero_valid_video_count": (
            distribution.get(
                0,
                0,
            )
        ),
        "temporal_plan_sha256": (
            temporal_fingerprint[
                "sha256"
            ]
        ),
        "clean_geometry_sha256": (
            geometry_fingerprint[
                "sha256"
            ]
        ),
    }

    for (
        field_name,
        expected_value,
    ) in expected_exact.items():
        if (
            summary.get(
                field_name
            )
            != expected_value
        ):
            raise ValueError(
                "Clean-geometry summary mismatch "
                "for {}.".format(
                    field_name
                )
            )

    expect_number(
        summary.get(
            "alignment_scale"
        ),
        CLEAN_GEOMETRY_ALIGNMENT_SCALE,
        field_name=(
            "clean_geometry.alignment_scale"
        ),
    )

    valid_record_count = (
        status_counts.get(
            "valid",
            0,
        )
    )
    invalid_record_count = (
        status_counts.get(
            "invalid",
            0,
        )
    )

    if (
        valid_record_count
        + invalid_record_count
        != CLEAN_GEOMETRY_RECORDS
    ):
        raise ValueError(
            "Clean-geometry valid/invalid counts "
            "do not sum to nominal positions."
        )

    return {
        "geometry": geometry_fingerprint,
        "summary": (
            artifact_fingerprint(
                summary_path
            )
        ),
        "predictor": predictor_fingerprint,
        "base_video_count": (
            EXPECTED_PROCESSING_RECORDS
        ),
        "nominal_position_count": (
            CLEAN_GEOMETRY_RECORDS
        ),
        "valid_position_count": (
            valid_record_count
        ),
        "invalid_position_count": (
            invalid_record_count
        ),
        "complete_32_video_count": (
            distribution.get(
                TEMPORAL_FRAME_BUDGET,
                0,
            )
        ),
        "incomplete_video_count": (
            EXPECTED_PROCESSING_RECORDS
            - distribution.get(
                TEMPORAL_FRAME_BUDGET,
                0,
            )
        ),
        "zero_valid_video_count": (
            distribution.get(
                0,
                0,
            )
        ),
        "failure_stages": dict(
            sorted(
                failure_stages.items()
            )
        ),
        "valid_count_distribution": (
            distribution_json
        ),
    }


def validate_threshold_artifact(
    artifact: Dict[str, Any],
    *,
    analysis_path: Path,
    checkpoint_hashes: Dict[
        str,
        str,
    ],
) -> Dict[str, Any]:
    if (
        artifact.get(
            "artifact_type"
        )
        != "primary_operating_thresholds"
    ):
        raise ValueError(
            "Unexpected primary threshold "
            "artifact type."
        )

    expect_number(
        artifact.get(
            "target_fpr"
        ),
        0.05,
        field_name="target_fpr",
    )

    analysis_input = artifact.get(
        "analysis_input"
    )

    if not isinstance(
        analysis_input,
        dict,
    ):
        raise ValueError(
            "Threshold artifact has no "
            "analysis_input provenance."
        )

    current_analysis_hash = (
        sha256_file(
            analysis_path
        )
    )

    if (
        analysis_input.get(
            "sha256"
        )
        != current_analysis_hash
    ):
        raise ValueError(
            "Threshold artifact analysis-input "
            "hash mismatch."
        )

    thresholds = artifact.get(
        "thresholds"
    )

    if not isinstance(
        thresholds,
        list,
    ):
        raise ValueError(
            "Threshold artifact has no "
            "threshold list."
        )

    if (
        artifact.get(
            "threshold_count"
        )
        != len(
            thresholds
        )
    ):
        raise ValueError(
            "threshold_count does not match "
            "the threshold list."
        )

    by_detector = {}

    for row in thresholds:
        detector = row.get(
            "detector"
        )

        if detector in by_detector:
            raise ValueError(
                "Duplicate primary threshold "
                "for detector {}.".format(
                    detector
                )
            )

        if detector not in EXPECTED_DETECTORS:
            raise ValueError(
                "Unexpected threshold detector: {}."
                .format(
                    detector
                )
            )

        if (
            row.get(
                "checkpoint_sha256"
            )
            != checkpoint_hashes[
                detector
            ]
        ):
            raise ValueError(
                "Threshold/checkpoint mismatch "
                "for detector {}.".format(
                    detector
                )
            )

        expected_metadata = {
            "purpose": (
                "primary_operating_point"
            ),

            "calibration_dataset": (
                "FaceForensics++"
            ),

            "calibration_role": (
                "validation"
            ),

            "calibration_condition": (
                "CLN"
            ),

            "calibration_class": (
                "real_only"
            ),

            "selection_rule": (
                "lowest_candidate_with_fpr_lte_target"
            ),

            "candidate_rule": (
                "unique_observed_real_scores_"
                "plus_nextafter_max"
            ),

            "tie_rule": (
                "lowest_numeric_threshold"
            ),

            "decision_rule": (
                "manipulated_if_score_gte_threshold"
            ),

            "score_direction": (
                "higher_means_more_manipulated"
            ),
        }

        for (
            field_name,
            expected_value,
        ) in expected_metadata.items():
            if (
                row.get(
                    field_name
                )
                != expected_value
            ):
                raise ValueError(
                    "Threshold metadata mismatch "
                    "for {} field {}.".format(
                        detector,
                        field_name,
                    )
                )

        expect_number(
            row.get(
                "target_fpr"
            ),
            0.05,
            field_name=(
                "{}.target_fpr".format(
                    detector
                )
            ),
        )

        realized_fpr = strict_number(
            row.get(
                "realized_fpr"
            ),
            field_name=(
                "{}.realized_fpr".format(
                    detector
                )
            ),
        )

        if realized_fpr > 0.05:
            raise ValueError(
                "Primary threshold violates "
                "target FPR for {}.".format(
                    detector
                )
            )

        threshold_value = (
            strict_number(
                row.get(
                    "threshold_value"
                ),
                field_name=(
                    "{}.threshold_value"
                    .format(
                        detector
                    )
                ),
            )
        )

        by_detector[
            detector
        ] = {
            "threshold_id": row.get(
                "threshold_id"
            ),

            "threshold_value": (
                threshold_value
            ),

            "realized_fpr": (
                realized_fpr
            ),
        }

    if (
        set(
            by_detector
        )
        != EXPECTED_DETECTORS
    ):
        raise ValueError(
            "Primary threshold detector set "
            "is incomplete."
        )

    return by_detector


def validate_threshold_decisions(
    *,
    decision_records: List[
        Dict[str, Any]
    ],
    threshold_artifact: Dict[
        str,
        Any,
    ],
):
    threshold_index = {
        row[
            "detector"
        ]: row
        for row in threshold_artifact[
            "thresholds"
        ]
    }

    for row in decision_records:
        detector = row[
            "detector"
        ]

        threshold = threshold_index[
            detector
        ]

        if (
            row[
                "checkpoint_sha256"
            ]
            != threshold[
                "checkpoint_sha256"
            ]
        ):
            raise ValueError(
                "Decision/checkpoint mismatch "
                "for detector {}.".format(
                    detector
                )
            )

        if (
            row[
                "threshold_id"
            ]
            != threshold[
                "threshold_id"
            ]
        ):
            raise ValueError(
                "Decision threshold ID mismatch "
                "for detector {}.".format(
                    detector
                )
            )

        if not math.isclose(
            float(
                row[
                    "threshold_value"
                ]
            ),
            float(
                threshold[
                    "threshold_value"
                ]
            ),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                "Decision threshold value mismatch "
                "for detector {}.".format(
                    detector
                )
            )

        if (
            row[
                "analysis_eligible"
            ]
            is True
        ):
            expected_decision = (
                1
                if float(
                    row[
                        "video_score"
                    ]
                )
                >= float(
                    threshold[
                        "threshold_value"
                    ]
                )
                else 0
            )

            if (
                row[
                    "decision_status"
                ]
                != "decided"
                or row[
                    "decision"
                ]
                != expected_decision
            ):
                raise ValueError(
                    "Frozen-threshold decision "
                    "mismatch for {}.".format(
                        analysis_record_key(
                            row
                        )
                    )
                )

        else:
            if (
                row[
                    "decision_status"
                ]
                != "not_scored"
                or row[
                    "decision"
                ]
                is not None
            ):
                raise ValueError(
                    "Invalid analysis record received "
                    "a binary decision."
                )


def validate_decision_summary(
    summary: Dict[str, Any],
    *,
    analysis_path: Path,
    thresholds_path: Path,
    decisions_path: Path,
    analysis_count: int,
    decision_count: int,
):
    if (
        summary.get(
            "artifact_type"
        )
        != "frozen_threshold_decisions"
    ):
        raise ValueError(
            "Unexpected decision-summary "
            "artifact type."
        )

    if summary.get(
        "status"
    ) != "passed":
        raise ValueError(
            "Decision summary is not passed."
        )

    if (
        summary.get(
            "analysis_record_count"
        )
        != analysis_count
    ):
        raise ValueError(
            "Decision-summary analysis count "
            "mismatch."
        )

    if (
        summary.get(
            "decision_record_count"
        )
        != decision_count
    ):
        raise ValueError(
            "Decision-summary output count "
            "mismatch."
        )

    expected_hashes = {
        "analysis_input_sha256": (
            sha256_file(
                analysis_path
            )
        ),

        "threshold_artifact_sha256": (
            sha256_file(
                thresholds_path
            )
        ),

        "decision_output_sha256": (
            sha256_file(
                decisions_path
            )
        ),
    }

    for (
        field_name,
        expected_hash,
    ) in expected_hashes.items():
        if (
            summary.get(
                field_name
            )
            != expected_hash
        ):
            raise ValueError(
                "Decision-summary hash mismatch "
                "for {}.".format(
                    field_name
                )
            )


def validate_source_artifacts(
    source_artifacts: Any,
    *,
    expected_paths: Dict[
        str,
        Path,
    ],
    context: str,
):
    if not isinstance(
        source_artifacts,
        dict,
    ):
        raise ValueError(
            "{} has no source_artifacts map."
            .format(
                context
            )
        )

    for (
        key,
        path,
    ) in expected_paths.items():
        entry = source_artifacts.get(
            key
        )

        if not isinstance(
            entry,
            dict,
        ):
            raise ValueError(
                "{} is missing source artifact {}."
                .format(
                    context,
                    key,
                )
            )

        expected_hash = sha256_file(
            path
        )

        if (
            entry.get(
                "sha256"
            )
            != expected_hash
        ):
            raise ValueError(
                "{} source hash mismatch "
                "for {}.".format(
                    context,
                    key,
                )
            )


def validate_metric_results(
    artifact: Dict[str, Any],
    *,
    analysis_path: Path,
    decisions_path: Path,
    thresholds_path: Path,
):
    if (
        artifact.get(
            "artifact_type"
        )
        != "metric_results"
    ):
        raise ValueError(
            "Unexpected metric-results artifact."
        )

    validate_source_artifacts(
        artifact.get(
            "source_artifacts"
        ),
        expected_paths={
            "analysis_jsonl": (
                analysis_path
            ),

            "decisions_jsonl": (
                decisions_path
            ),

            "primary_thresholds_json": (
                thresholds_path
            ),
        },
        context="metric_results",
    )


def validate_bootstrap_configuration(
    configuration: Any,
):
    if not isinstance(
        configuration,
        dict,
    ):
        raise ValueError(
            "Missing bootstrap configuration."
        )

    expected_exact = {
        "replicates": 2000,
        "seed": 1024,

        "ci_method": (
            "percentile_linear_interpolation"
        ),

        "rng_stream_policy": (
            "deterministic_named_streams"
        ),

        "paired_clean_degraded_sampling": True,

        "threshold_reestimated_each_replicate": True,

        "production_configuration": True,
    }

    for (
        field_name,
        expected_value,
    ) in expected_exact.items():
        if (
            configuration.get(
                field_name
            )
            != expected_value
        ):
            raise ValueError(
                "Bootstrap configuration mismatch "
                "for {}.".format(
                    field_name
                )
            )

    expect_number(
        configuration.get(
            "ci_level"
        ),
        0.95,
        field_name="bootstrap.ci_level",
    )

    expect_number(
        configuration.get(
            "primary_target_fpr"
        ),
        0.05,
        field_name=(
            "bootstrap.primary_target_fpr"
        ),
    )


def validate_bootstrap_results(
    artifact: Dict[str, Any],
    *,
    analysis_path: Path,
    decisions_path: Path,
    metric_results_path: Path,
):
    if (
        artifact.get(
            "artifact_type"
        )
        != "bootstrap_results"
    ):
        raise ValueError(
            "Unexpected bootstrap-results artifact."
        )

    validate_source_artifacts(
        artifact.get(
            "source_artifacts"
        ),
        expected_paths={
            "analysis_jsonl": (
                analysis_path
            ),

            "decisions_jsonl": (
                decisions_path
            ),

            "metric_results_json": (
                metric_results_path
            ),
        },
        context="bootstrap_results",
    )

    validate_bootstrap_configuration(
        artifact.get(
            "bootstrap_configuration"
        )
    )


def validate_export_outputs(
    summary: Dict[str, Any],
    *,
    exports_dir: Path,
) -> Dict[str, Any]:
    if (
        summary.get(
            "artifact_type"
        )
        != "analysis_export_summary"
    ):
        raise ValueError(
            "Unexpected analysis-export "
            "summary artifact."
        )

    if (
        summary.get(
            "status"
        )
        != "passed"
    ):
        raise ValueError(
            "Analysis export summary is not passed."
        )

    outputs = summary.get(
        "outputs"
    )

    if not isinstance(
        outputs,
        dict,
    ):
        raise ValueError(
            "Analysis export summary has no outputs."
        )

    required_outputs = {
        "absolute_metrics_csv",
        "paired_degradation_metrics_csv",
        "diagnostic_thresholds_csv",
        "score_records_csv",
        "compact_results_json",
        "analysis_provenance_json",
    }

    missing = (
        required_outputs
        - set(
            outputs
        )
    )

    if missing:
        raise ValueError(
            "Missing exported artifacts: {}."
            .format(
                sorted(
                    missing
                )
            )
        )

    verified = {}

    resolved_export_dir = (
        exports_dir
        .expanduser()
        .resolve()
    )

    for name in sorted(
        required_outputs
    ):
        entry = outputs[
            name
        ]

        if not isinstance(
            entry,
            dict,
        ):
            raise ValueError(
                "Invalid export-summary entry: {}."
                .format(
                    name
                )
            )

        path = Path(
            entry[
                "path"
            ]
        ).expanduser().resolve()

        if (
            path.parent
            != resolved_export_dir
        ):
            raise ValueError(
                "Export artifact {} points outside "
                "the expected output directory."
                .format(
                    name
                )
            )

        fingerprint = (
            artifact_fingerprint(
                path
            )
        )

        if (
            fingerprint[
                "sha256"
            ]
            != entry.get(
                "sha256"
            )
        ):
            raise ValueError(
                "Export SHA-256 mismatch for {}."
                .format(
                    name
                )
            )

        if (
            fingerprint[
                "size_bytes"
            ]
            != entry.get(
                "size_bytes"
            )
        ):
            raise ValueError(
                "Export size mismatch for {}."
                .format(
                    name
                )
            )

        verified[
            name
        ] = fingerprint

    return verified


def validate_tracked_files(
    provenance: Dict[str, Any],
) -> int:
    tracked = provenance.get(
        "tracked_files"
    )

    if not isinstance(
        tracked,
        list,
    ):
        raise ValueError(
            "Analysis provenance has no "
            "tracked-file list."
        )

    for entry in tracked:
        if not isinstance(
            entry,
            dict,
        ):
            raise ValueError(
                "Invalid tracked-file entry."
            )

        relative_path = entry.get(
            "path"
        )

        if (
            type(relative_path) is not str
            or not relative_path
        ):
            raise ValueError(
                "Tracked file has invalid path."
            )

        path = Path(
            relative_path
        )

        if (
            path.is_absolute()
            or ".." in path.parts
        ):
            raise ValueError(
                "Unsafe tracked-file path: {}."
                .format(
                    relative_path
                )
            )

        absolute_path = (
            REPO_ROOT
            / path
        )

        fingerprint = (
            artifact_fingerprint(
                absolute_path
            )
        )

        if (
            fingerprint[
                "sha256"
            ]
            != entry.get(
                "sha256"
            )
        ):
            raise ValueError(
                "Tracked implementation/config "
                "hash mismatch: {}.".format(
                    relative_path
                )
            )

        if (
            fingerprint[
                "size_bytes"
            ]
            != entry.get(
                "size_bytes"
            )
        ):
            raise ValueError(
                "Tracked implementation/config "
                "size mismatch: {}.".format(
                    relative_path
                )
            )

    if not tracked:
        raise ValueError(
            "Tracked-file list is empty."
        )

    return len(
        tracked
    )


def validate_analysis_provenance(
    provenance: Dict[str, Any],
    *,
    analysis_path: Path,
    decisions_path: Path,
    metric_results_path: Path,
    bootstrap_results_path: Path,
    thresholds_path: Path,
    bootstrap_results: Dict[str, Any],
) -> Dict[str, Any]:
    if (
        provenance.get(
            "artifact_type"
        )
        != "analysis_provenance"
    ):
        raise ValueError(
            "Unexpected analysis provenance "
            "artifact."
        )

    validate_source_artifacts(
        provenance.get(
            "direct_inputs"
        ),
        expected_paths={
            "analysis_jsonl": (
                analysis_path
            ),

            "decisions_jsonl": (
                decisions_path
            ),

            "metric_results_json": (
                metric_results_path
            ),

            "bootstrap_results_json": (
                bootstrap_results_path
            ),
        },
        context=(
            "analysis_provenance.direct_inputs"
        ),
    )

    validate_source_artifacts(
        provenance.get(
            "metric_result_sources"
        ),
        expected_paths={
            "analysis_jsonl": (
                analysis_path
            ),

            "decisions_jsonl": (
                decisions_path
            ),

            "primary_thresholds_json": (
                thresholds_path
            ),
        },
        context=(
            "analysis_provenance."
            "metric_result_sources"
        ),
    )

    validate_source_artifacts(
        provenance.get(
            "bootstrap_result_sources"
        ),
        expected_paths={
            "analysis_jsonl": (
                analysis_path
            ),

            "decisions_jsonl": (
                decisions_path
            ),

            "metric_results_json": (
                metric_results_path
            ),
        },
        context=(
            "analysis_provenance."
            "bootstrap_result_sources"
        ),
    )

    if (
        provenance.get(
            "bootstrap_configuration"
        )
        != bootstrap_results.get(
            "bootstrap_configuration"
        )
    ):
        raise ValueError(
            "Export provenance bootstrap "
            "configuration differs from the "
            "bootstrap artifact."
        )

    validate_bootstrap_configuration(
        provenance.get(
            "bootstrap_configuration"
        )
    )

    runtime = provenance.get(
        "runtime"
    )

    if not isinstance(
        runtime,
        dict,
    ):
        raise ValueError(
            "Analysis provenance has no "
            "runtime metadata."
        )

    for field_name in (
        "python",
        "python_implementation",
        "numpy",
        "platform",
    ):
        if not runtime.get(
            field_name
        ):
            raise ValueError(
                "Analysis provenance runtime "
                "field is missing: {}.".format(
                    field_name
                )
            )

    git = provenance.get(
        "git"
    )

    if not isinstance(
        git,
        dict,
    ):
        raise ValueError(
            "Analysis provenance has no "
            "Git metadata."
        )

    tracked_count = (
        validate_tracked_files(
            provenance
        )
    )

    return {
        "recorded_runtime": runtime,
        "recorded_git": git,

        "tracked_file_count": (
            tracked_count
        ),
    }


def validate_mode_contract(
    *,
    mode: str,
    inference_records: List[
        Dict[str, Any]
    ],
    analysis_records: List[
        Dict[str, Any]
    ],
    decision_records: List[
        Dict[str, Any]
    ],
    export_summary: Dict[str, Any],
):
    detectors = {
        row[
            "detector"
        ]
        for row in analysis_records
    }

    conditions = {
        row[
            "condition"
        ]
        for row in analysis_records
    }

    if (
        detectors
        != EXPECTED_DETECTORS
    ):
        raise ValueError(
            "Audit detector set mismatch: {}."
            .format(
                sorted(
                    detectors
                )
            )
        )

    if mode == "smoke":
        if not conditions:
            raise ValueError(
                "Smoke analysis has no conditions."
            )

        if not conditions.issubset(
            EXPECTED_CONDITIONS
        ):
            raise ValueError(
                "Smoke analysis contains "
                "unsupported conditions."
            )

        return {
            "mode": "smoke",

            "detectors": sorted(
                detectors
            ),

            "conditions_observed": sorted(
                conditions
            ),

            "record_count": len(
                analysis_records
            ),
        }

    if (
        conditions
        != EXPECTED_CONDITIONS
    ):
        raise ValueError(
            "Production analysis does not contain "
            "all five study conditions."
        )

    expected_count = (
        EXPECTED_PRODUCTION_RECORDS
    )

    for (
        name,
        rows,
    ) in (
        (
            "inference",
            inference_records,
        ),
        (
            "analysis",
            analysis_records,
        ),
        (
            "decision",
            decision_records,
        ),
    ):
        if (
            len(
                rows
            )
            != expected_count
        ):
            raise ValueError(
                "Production {} record count "
                "mismatch: {} != {}.".format(
                    name,
                    len(
                        rows
                    ),
                    expected_count,
                )
            )

    if (
        export_summary.get(
            "allow_empty_primary_metrics"
        )
        is not False
    ):
        raise ValueError(
            "Production export must not permit "
            "empty primary metrics."
        )

    if (
        export_summary.get(
            "absolute_metric_count",
            0,
        )
        <= 0
    ):
        raise ValueError(
            "Production export has no "
            "absolute metric results."
        )

    if (
        export_summary.get(
            "paired_degradation_count",
            0,
        )
        <= 0
    ):
        raise ValueError(
            "Production export has no paired "
            "degradation results."
        )

    return {
        "mode": "production",

        "detectors": sorted(
            detectors
        ),

        "conditions_observed": sorted(
            conditions
        ),

        "record_count": len(
            analysis_records
        ),
    }


def audit_implementation_files() -> Dict[
    str,
    Dict[str, Any],
]:
    output = {}

    for relative_path in (
        AUDIT_IMPLEMENTATION_FILES
    ):
        path = (
            REPO_ROOT
            / relative_path
        )

        output[
            relative_path
        ] = (
            artifact_fingerprint(
                path
            )
        )

    return output


def build_audit(
    args,
) -> Dict[str, Any]:
    checks = []

    #
    # --------------------------------------------------
    # Canonical manifest
    # --------------------------------------------------
    #
    manifest_rows = read_manifest_csv(
        args.study_manifest
    )

    manifest_index = (
        build_manifest_index(
            manifest_rows
        )
    )

    manifest_population = (
        validate_manifest_population(
            manifest_rows
        )
    )

    add_check(
        checks,
        "canonical_manifest",
        manifest_population,
    )

    #
    # --------------------------------------------------
    # Inference / analysis / decision schema
    # --------------------------------------------------
    #
    inference_records = read_jsonl(
        args.inference_jsonl
    )

    inference_count = (
        validate_inference_records(
            inference_records
        )
    )

    add_check(
        checks,
        "canonical_inference_schema",
        {
            "record_count": (
                inference_count
            ),
        },
    )

    analysis_records = read_jsonl(
        args.analysis_jsonl
    )

    analysis_count = (
        validate_analysis_records(
            analysis_records
        )
    )

    add_check(
        checks,
        "analysis_record_schema",
        {
            "record_count": (
                analysis_count
            ),
        },
    )

    validate_inference_analysis_chain(
        inference_records=(
            inference_records
        ),
        analysis_records=(
            analysis_records
        ),
    )

    add_check(
        checks,
        "inference_to_analysis_chain",
    )

    validate_manifest_analysis_chain(
        manifest_index=(
            manifest_index
        ),
        inference_records=(
            inference_records
        ),
        analysis_records=(
            analysis_records
        ),
    )

    add_check(
        checks,
        "manifest_to_analysis_chain",
    )

    decision_records = read_jsonl(
        args.decisions_jsonl
    )

    decision_count = (
        validate_decision_records(
            decision_records
        )
    )

    validate_cross_stage_identity(
        analysis_records=(
            analysis_records
        ),
        decision_records=(
            decision_records
        ),
    )

    add_check(
        checks,
        "analysis_to_decision_chain",
        {
            "record_count": (
                decision_count
            ),
        },
    )

    #
    # --------------------------------------------------
    # Final detector checkpoints
    # --------------------------------------------------
    #
    checkpoint_hashes = (
        checkpoint_hashes_from_analysis(
            analysis_records
        )
    )

    add_check(
        checks,
        "checkpoint_files",
        {
            "sha256_by_detector": (
                checkpoint_hashes
            ),
        },
    )

    #
    # --------------------------------------------------
    # Training provenance
    # --------------------------------------------------
    #
    training_provenance = (
        validate_training_provenance(
            metadata_paths=(
                args.training_metadata_json
            ),
            checkpoint_hashes=(
                checkpoint_hashes
            ),
        )
    )

    add_check(
        checks,
        "training_selection_provenance",
        {
            detector: {
                "selected_epoch": (
                    training_provenance[
                        detector
                    ][
                        "selected_epoch"
                    ]
                ),

                "selection_value": (
                    training_provenance[
                        detector
                    ][
                        "selection_value"
                    ]
                ),

                "optimizer_steps_completed": (
                    training_provenance[
                        detector
                    ][
                        "optimizer_steps_completed"
                    ]
                ),
            }
            for detector in sorted(
                training_provenance
            )
        },
    )

    #
    # --------------------------------------------------
    # Processing provenance
    # --------------------------------------------------
    #
    processing_provenance = (
        validate_processing_provenance(
            provenance_path=(
                args.processing_provenance_json
            ),
            processing_manifest_path=(
                args.processing_manifest
            ),
            canonical_manifest_rows=(
                manifest_rows
            ),
        )
    )

    add_check(
        checks,
        "processing_provenance",
        {
            "processing_manifest_records": (
                processing_provenance[
                    "population"
                ][
                    "records"
                ]
            ),

            "canonical_universe_match": (
                processing_provenance[
                    "canonical_universe_match"
                ]
            ),

            "implementation_file_count": (
                processing_provenance[
                    "implementation_file_count"
                ]
            ),
        },
    )

    processing_qc = (
        validate_processing_qc(
            summary_paths=(
                args.processing_qc_summary_json
            ),
            mode=args.mode,
        )
    )

    add_check(
        checks,
        "processing_qc",
        {
            condition: {
                "selected_videos": (
                    processing_qc[
                        condition
                    ][
                        "selected_videos"
                    ]
                ),

                "status_counts": (
                    processing_qc[
                        condition
                    ][
                        "status_counts"
                    ]
                ),
            }
            for condition in (
                EXPECTED_PROCESSED_CONDITIONS
            )
        },
    )

    #
    # --------------------------------------------------
    # Frozen temporal plan
    # --------------------------------------------------
    #
    temporal_plan = (
        validate_temporal_plan_artifacts(
            plan_path=(
                args.temporal_plan_csv
            ),
            summary_path=(
                args.temporal_plan_summary_json
            ),
            processing_manifest_path=(
                args.processing_manifest
            ),
        )
    )

    add_check(
        checks,
        "temporal_plan_provenance",
        {
            "base_video_count": (
                temporal_plan[
                    "base_video_count"
                ]
            ),
            "target_frame_budget": (
                temporal_plan[
                    "target_frame_budget"
                ]
            ),
            "sampling_method": (
                temporal_plan[
                    "sampling_method"
                ]
            ),
            "minimum_decoded_frame_count": (
                temporal_plan[
                    "minimum_decoded_frame_count"
                ]
            ),
            "maximum_decoded_frame_count": (
                temporal_plan[
                    "maximum_decoded_frame_count"
                ]
            ),
            "sha256": (
                temporal_plan[
                    "plan"
                ][
                    "sha256"
                ]
            ),
        },
    )

    #
    # --------------------------------------------------
    # Frozen clean-reference geometry
    # --------------------------------------------------
    #
    clean_geometry = (
        validate_clean_geometry_artifacts(
            geometry_path=(
                args.clean_geometry_csv
            ),
            summary_path=(
                args.clean_geometry_summary_json
            ),
            temporal_plan_path=(
                args.temporal_plan_csv
            ),
            temporal_plan_details=(
                temporal_plan
            ),
        )
    )

    add_check(
        checks,
        "clean_geometry_provenance",
        {
            "base_video_count": (
                clean_geometry[
                    "base_video_count"
                ]
            ),
            "nominal_position_count": (
                clean_geometry[
                    "nominal_position_count"
                ]
            ),
            "valid_position_count": (
                clean_geometry[
                    "valid_position_count"
                ]
            ),
            "invalid_position_count": (
                clean_geometry[
                    "invalid_position_count"
                ]
            ),
            "complete_32_video_count": (
                clean_geometry[
                    "complete_32_video_count"
                ]
            ),
            "zero_valid_video_count": (
                clean_geometry[
                    "zero_valid_video_count"
                ]
            ),
            "sha256": (
                clean_geometry[
                    "geometry"
                ][
                    "sha256"
                ]
            ),
        },
    )

        #
    # --------------------------------------------------
    # Frozen runtime/environment snapshot
    # --------------------------------------------------
    #
    environment_snapshot = (
        validate_environment_snapshot(
            args.environment_snapshot_json
        )
    )

    add_check(
        checks,
        "environment_runtime_snapshot",
        {
            "created_utc": (
                environment_snapshot[
                    "created_utc"
                ]
            ),

            "python": (
                environment_snapshot[
                    "python"
                ][
                    "version_short"
                ]
            ),

            "numpy": (
                environment_snapshot[
                    "libraries"
                ][
                    "numpy"
                ]
            ),

            "pytorch": (
                environment_snapshot[
                    "libraries"
                ][
                    "pytorch"
                ]
            ),

            "opencv": (
                environment_snapshot[
                    "libraries"
                ][
                    "opencv"
                ]
            ),

            "pyyaml": (
                environment_snapshot[
                    "libraries"
                ][
                    "pyyaml"
                ]
            ),

            "cuda_available": (
                environment_snapshot[
                    "cuda"
                ][
                    "available"
                ]
            ),

            "pytorch_cuda": (
                environment_snapshot[
                    "cuda"
                ][
                    "pytorch_cuda"
                ]
            ),

            "cudnn_version": (
                environment_snapshot[
                    "cuda"
                ][
                    "cudnn_version"
                ]
            ),

            "device_count": (
                environment_snapshot[
                    "cuda"
                ][
                    "device_count"
                ]
            ),

            "git_commit": (
                environment_snapshot[
                    "repository"
                ][
                    "commit"
                ]
            ),

            "git_branch": (
                environment_snapshot[
                    "repository"
                ][
                    "branch"
                ]
            ),

            "working_tree_dirty": (
                environment_snapshot[
                    "repository"
                ][
                    "working_tree_dirty"
                ]
            ),

            "sha256": (
                environment_snapshot[
                    "snapshot"
                ][
                    "sha256"
                ]
            ),
        },
    )

    #
    # --------------------------------------------------
    # Thresholds / decisions
    # --------------------------------------------------
    #
    threshold_artifact = read_json(
        args.thresholds_json
    )

    threshold_summary = (
        validate_threshold_artifact(
            threshold_artifact,
            analysis_path=(
                args.analysis_jsonl
            ),
            checkpoint_hashes=(
                checkpoint_hashes
            ),
        )
    )

    validate_threshold_decisions(
        decision_records=(
            decision_records
        ),
        threshold_artifact=(
            threshold_artifact
        ),
    )

    add_check(
        checks,
        "primary_threshold_and_decisions",
        {
            "thresholds": (
                threshold_summary
            ),
        },
    )

    decision_summary = read_json(
        args.decisions_summary_json
    )

    validate_decision_summary(
        decision_summary,
        analysis_path=(
            args.analysis_jsonl
        ),
        thresholds_path=(
            args.thresholds_json
        ),
        decisions_path=(
            args.decisions_jsonl
        ),
        analysis_count=(
            analysis_count
        ),
        decision_count=(
            decision_count
        ),
    )

    add_check(
        checks,
        "decision_artifact_provenance",
    )

    #
    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    #
    metric_results = read_json(
        args.metric_results_json
    )

    validate_metric_results(
        metric_results,
        analysis_path=(
            args.analysis_jsonl
        ),
        decisions_path=(
            args.decisions_jsonl
        ),
        thresholds_path=(
            args.thresholds_json
        ),
    )

    add_check(
        checks,
        "metric_result_provenance",
    )

    #
    # --------------------------------------------------
    # Bootstrap
    # --------------------------------------------------
    #
    bootstrap_results = read_json(
        args.bootstrap_results_json
    )

    validate_bootstrap_results(
        bootstrap_results,
        analysis_path=(
            args.analysis_jsonl
        ),
        decisions_path=(
            args.decisions_jsonl
        ),
        metric_results_path=(
            args.metric_results_json
        ),
    )

    add_check(
        checks,
        "bootstrap_provenance_and_configuration",
        {
            "configuration": (
                bootstrap_results[
                    "bootstrap_configuration"
                ]
            ),
        },
    )

    #
    # --------------------------------------------------
    # Exports / analysis provenance
    # --------------------------------------------------
    #
    export_summary_path = (
        args.exports_dir
        / "analysis_export_summary.json"
    )

    export_summary = read_json(
        export_summary_path
    )

    verified_outputs = (
        validate_export_outputs(
            export_summary,
            exports_dir=(
                args.exports_dir
            ),
        )
    )

    provenance_path = (
        args.exports_dir
        / "analysis_provenance.json"
    )

    provenance = read_json(
        provenance_path
    )

    provenance_details = (
        validate_analysis_provenance(
            provenance,
            analysis_path=(
                args.analysis_jsonl
            ),
            decisions_path=(
                args.decisions_jsonl
            ),
            metric_results_path=(
                args.metric_results_json
            ),
            bootstrap_results_path=(
                args.bootstrap_results_json
            ),
            thresholds_path=(
                args.thresholds_json
            ),
            bootstrap_results=(
                bootstrap_results
            ),
        )
    )

    add_check(
        checks,
        "analysis_exports_and_provenance",
        {
            "verified_output_count": (
                len(
                    verified_outputs
                )
            ),

            "tracked_file_count": (
                provenance_details[
                    "tracked_file_count"
                ]
            ),
        },
    )

    #
    # --------------------------------------------------
    # Smoke / production contract
    # --------------------------------------------------
    #
    mode_details = (
        validate_mode_contract(
            mode=args.mode,
            inference_records=(
                inference_records
            ),
            analysis_records=(
                analysis_records
            ),
            decision_records=(
                decision_records
            ),
            export_summary=(
                export_summary
            ),
        )
    )

    add_check(
        checks,
        "study_mode_contract",
        mode_details,
    )

    artifacts = {
        "study_manifest": (
            artifact_fingerprint(
                args.study_manifest
            )
        ),

        "processing_manifest": (
            artifact_fingerprint(
                args.processing_manifest
            )
        ),

        "processing_provenance": (
            artifact_fingerprint(
                args.processing_provenance_json
            )
        ),

        "temporal_plan": (
            artifact_fingerprint(
                args.temporal_plan_csv
            )
        ),

        "temporal_plan_summary": (
            artifact_fingerprint(
                args.temporal_plan_summary_json
            )
        ),

        "clean_geometry": (
            artifact_fingerprint(
                args.clean_geometry_csv
            )
        ),

        "clean_geometry_summary": (
            artifact_fingerprint(
                args.clean_geometry_summary_json
            )
        ),

        "environment_snapshot": (
            artifact_fingerprint(
                args.environment_snapshot_json
            )
        ),

        "canonical_inference": (
            artifact_fingerprint(
                args.inference_jsonl
            )
        ),

        "analysis_records": (
            artifact_fingerprint(
                args.analysis_jsonl
            )
        ),

        "primary_thresholds": (
            artifact_fingerprint(
                args.thresholds_json
            )
        ),

        "decisions": (
            artifact_fingerprint(
                args.decisions_jsonl
            )
        ),

        "decisions_summary": (
            artifact_fingerprint(
                args.decisions_summary_json
            )
        ),

        "metric_results": (
            artifact_fingerprint(
                args.metric_results_json
            )
        ),

        "bootstrap_results": (
            artifact_fingerprint(
                args.bootstrap_results_json
            )
        ),

        "analysis_export_summary": (
            artifact_fingerprint(
                export_summary_path
            )
        ),

        "analysis_provenance": (
            artifact_fingerprint(
                provenance_path
            )
        ),
    }

    checkpoints = {
        detector: (
            artifact_fingerprint(
                CHECKPOINT_PATHS[
                    detector
                ]
            )
        )
        for detector in sorted(
            EXPECTED_DETECTORS
        )
    }

    return {
        "schema_version": 1,

        "artifact_type": (
            "reproducibility_audit"
        ),

        "status": "passed",

        "mode": args.mode,

        "study": {
            "detectors": sorted(
                EXPECTED_DETECTORS
            ),

            "expected_conditions": sorted(
                EXPECTED_CONDITIONS
            ),

            "observed_conditions": sorted(
                {
                    row[
                        "condition"
                    ]
                    for row in analysis_records
                }
            ),

            "canonical_manifest_records": (
                len(
                    manifest_rows
                )
            ),

            "processing_manifest_records": (
                EXPECTED_PROCESSING_RECORDS
            ),

            "temporal_plan_records": (
                temporal_plan[
                    "base_video_count"
                ]
            ),

            "clean_geometry_records": (
                clean_geometry[
                    "nominal_position_count"
                ]
            ),

            "clean_geometry_valid_records": (
                clean_geometry[
                    "valid_position_count"
                ]
            ),

            "clean_geometry_invalid_records": (
                clean_geometry[
                    "invalid_position_count"
                ]
            ),

            "inference_records": (
                len(
                    inference_records
                )
            ),

            "analysis_records": (
                len(
                    analysis_records
                )
            ),

            "decision_records": (
                len(
                    decision_records
                )
            ),
        },

        "checks": checks,

        "training_provenance": (
            training_provenance
        ),

        "processing_provenance": (
            processing_provenance
        ),

        "processing_qc": (
            processing_qc
        ),

        "temporal_plan": (
            {
                key: value
                for (
                    key,
                    value,
                ) in temporal_plan.items()
                if key != "index"
            }
        ),

        "clean_geometry": (
            clean_geometry
        ),

        "environment_snapshot": (
            environment_snapshot
        ),

        "bootstrap_configuration": (
            bootstrap_results[
                "bootstrap_configuration"
            ]
        ),

        "runtime": (
            current_runtime()
        ),

        "git": (
            current_git_state()
        ),

        "recorded_analysis_provenance": {
            "runtime": (
                provenance_details[
                    "recorded_runtime"
                ]
            ),

            "git": (
                provenance_details[
                    "recorded_git"
                ]
            ),
        },

        "artifacts": artifacts,

        "checkpoints": checkpoints,

        "export_outputs": (
            verified_outputs
        ),

        "audit_implementation": (
            audit_implementation_files()
        ),

        "summary": {
            "checks_total": len(
                checks
            ),

            "checks_passed": len(
                checks
            ),

            "checks_failed": 0,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the final cross-stage reproducibility "
            "audit over canonical study artifacts."
        )
    )

    parser.add_argument(
        "--mode",
        choices=(
            "smoke",
            "production",
        ),
        default="smoke",
    )

    parser.add_argument(
        "--study-manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--processing-manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--processing-provenance-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--processing-qc-summary-json",
        required=True,
        action="append",
        type=Path,
        help=(
            "Repeat once for each of RSZ, BLR, "
            "H40, and PLT."
        ),
    )

    parser.add_argument(
        "--training-metadata-json",
        required=True,
        action="append",
        type=Path,
        help=(
            "Repeat once for each selected "
            "Xception, UCF, and SPSL run."
        ),
    )

    parser.add_argument(
        "--inference-jsonl",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--analysis-jsonl",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--thresholds-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--decisions-jsonl",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--decisions-summary-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--metric-results-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--bootstrap-results-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--exports-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    scalar_path_fields = (
        "study_manifest",
        "processing_manifest",
        "processing_provenance_json",
        "inference_jsonl",
        "analysis_jsonl",
        "thresholds_json",
        "decisions_jsonl",
        "decisions_summary_json",
        "metric_results_json",
        "bootstrap_results_json",
        "exports_dir",
        "output_json",
    )

    for field_name in (
        scalar_path_fields
    ):
        value = getattr(
            args,
            field_name,
        )

        setattr(
            args,
            field_name,
            value.expanduser().resolve(),
        )

    study_root = (
        args.processing_manifest
        .parent
        .parent
    )

    args.temporal_plan_csv = (
        study_root
        / "evaluation"
        / "temporal_plan.csv"
    ).resolve()

    args.temporal_plan_summary_json = (
        study_root
        / "evaluation"
        / "temporal_plan_summary.json"
    ).resolve()

    args.clean_geometry_csv = (
        study_root
        / "evaluation"
        / "clean_geometry.csv"
    ).resolve()

    args.clean_geometry_summary_json = (
        study_root
        / "evaluation"
        / "clean_geometry_summary.json"
    ).resolve()

    args.environment_snapshot_json = (
        study_root
        / "provenance"
        / "environment_snapshot.json"
    ).resolve()

    args.training_metadata_json = [
        path.expanduser().resolve()
        for path in (
            args.training_metadata_json
        )
    ]

    args.processing_qc_summary_json = [
        path.expanduser().resolve()
        for path in (
            args.processing_qc_summary_json
        )
    ]

    try:
        audit = build_audit(
            args
        )

    except Exception as exc:
        failed_audit = {
            "schema_version": 1,

            "artifact_type": (
                "reproducibility_audit"
            ),

            "status": "failed",

            "mode": args.mode,

            "error": {
                "type": (
                    type(
                        exc
                    ).__name__
                ),

                "message": str(
                    exc
                ),
            },

            "runtime": (
                current_runtime()
            ),

            "git": (
                current_git_state()
            ),

            "summary": {
                "checks_failed": 1,
            },
        }

        write_json(
            args.output_json,
            failed_audit,
        )

        raise

    write_json(
        args.output_json,
        audit,
    )

    print(
        "REPRODUCIBILITY AUDIT"
    )

    print(
        "  mode:                       {}".format(
            audit[
                "mode"
            ]
        )
    )

    print(
        "  canonical manifest records: {}".format(
            audit[
                "study"
            ][
                "canonical_manifest_records"
            ]
        )
    )

    print(
        "  processing manifest records:{}".format(
            audit[
                "study"
            ][
                "processing_manifest_records"
            ]
        )
    )

    print(
        "  temporal plan records:      {}".format(
            audit[
                "study"
            ][
                "temporal_plan_records"
            ]
        )
    )

    print(
        "  clean geometry records:     {}".format(
            audit[
                "study"
            ][
                "clean_geometry_records"
            ]
        )
    )

    print(
        "  inference records:          {}".format(
            audit[
                "study"
            ][
                "inference_records"
            ]
        )
    )

    print(
        "  analysis records:           {}".format(
            audit[
                "study"
            ][
                "analysis_records"
            ]
        )
    )

    print(
        "  decision records:           {}".format(
            audit[
                "study"
            ][
                "decision_records"
            ]
        )
    )

    print(
        "  checks passed:              {}/{}".format(
            audit[
                "summary"
            ][
                "checks_passed"
            ],
            audit[
                "summary"
            ][
                "checks_total"
            ],
        )
    )

    print(
        "  manifest/role isolation:    PASSED"
    )

    print(
        "  training selection lineage: PASSED"
    )

    print(
        "  processing provenance:      PASSED"
    )

    print(
        "  processing QC:              PASSED"
    )

    print(
        "  temporal plan provenance:   PASSED"
    )

    print(
        "  clean geometry provenance:  PASSED"
    )

    print(
        "  environment/runtime snapshot:PASSED"
    )

    print(
        "  inference/analysis chain:   PASSED"
    )

    print(
        "  analysis/decision chain:    PASSED"
    )

    print(
        "  checkpoint fingerprints:    PASSED"
    )

    print(
        "  threshold provenance:       PASSED"
    )

    print(
        "  metric provenance:          PASSED"
    )

    print(
        "  bootstrap provenance:       PASSED"
    )

    print(
        "  export fingerprints:        PASSED"
    )

    print(
        "  tracked code/config hashes: PASSED"
    )

    print(
        "  output:                     {}".format(
            args.output_json
        )
    )

    print()

    print(
        "REPRODUCIBILITY AUDIT PASSED"
    )


if __name__ == "__main__":
    main()