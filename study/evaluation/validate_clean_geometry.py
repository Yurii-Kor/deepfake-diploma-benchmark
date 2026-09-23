import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from study.materialization.face_alignment import (
    ALIGNMENT_METHOD,
    ALIGNMENT_SCALE,
    DETECTOR_UPSAMPLE,
    LANDMARK_INDICES,
    OUTPUT_SIZE,
    estimate_similarity_matrix,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

STUDY_ROOT = (
    Path.home()
    / "deepfake_lab"
    / "study_data"
)

TEMPORAL_PLAN_PATH = (
    STUDY_ROOT
    / "evaluation"
    / "temporal_plan.csv"
)

CLEAN_GEOMETRY_PATH = (
    STUDY_ROOT
    / "evaluation"
    / "clean_geometry.csv"
)

SUMMARY_PATH = (
    STUDY_ROOT
    / "evaluation"
    / "clean_geometry_summary.json"
)

PREDICTOR_PATH = (
    PROJECT_ROOT
    / "preprocessing"
    / "dlib_tools"
    / "shape_predictor_81_face_landmarks.dat"
)

TRAINING_SUMMARY_PATH = (
    PROJECT_ROOT
    / "study"
    / "materialization"
    / "artifacts"
    / "frame_materialization_summary.json"
)

EXPECTED_BASE_VIDEOS = 1358
TARGET_FRAME_BUDGET = 32
EXPECTED_RECORDS = (
    EXPECTED_BASE_VIDEOS
    * TARGET_FRAME_BUDGET
)


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(
            "Required CSV does not exist: {}".format(
                path
            )
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(
            csv.DictReader(
                file
            )
        )


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(
            "Required JSON does not exist: {}".format(
                path
            )
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def sha256_file(path):
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


def is_sha256(value):
    if (
        not isinstance(
            value,
            str,
        )
        or len(
            value
        )
        != 64
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


def video_key(row):
    return (
        row["dataset"],
        row["role"],
        row["relative_source_path"],
    )


def frame_key(row):
    return (
        row["dataset"],
        row["role"],
        row["relative_source_path"],
        int(
            row[
                "temporal_position"
            ]
        ),
    )


def parse_json_cell(
    value,
    *,
    description,
):
    try:
        return json.loads(
            value
        )
    except json.JSONDecodeError as exc:
        raise AssertionError(
            "Invalid JSON in {}: {!r}".format(
                description,
                value,
            )
        ) from exc


def validate_training_contract(
    summary,
):
    training_summary = read_json(
        TRAINING_SUMMARY_PATH
    )

    current_predictor_hash = (
        sha256_file(
            PREDICTOR_PATH
        )
    )

    if (
        training_summary[
            "alignment_method"
        ]
        != ALIGNMENT_METHOD
    ):
        raise AssertionError(
            "Training alignment method differs "
            "from current implementation."
        )

    if (
        training_summary[
            "output_size"
        ]
        != OUTPUT_SIZE
    ):
        raise AssertionError(
            "Training alignment output size differs "
            "from current implementation."
        )

    if (
        training_summary[
            "predictor_sha256"
        ]
        != current_predictor_hash
    ):
        raise AssertionError(
            "Current dlib predictor differs from "
            "training materialization predictor."
        )

    if (
        summary[
            "alignment_method"
        ]
        != ALIGNMENT_METHOD
    ):
        raise AssertionError(
            "Clean-geometry alignment method mismatch."
        )

    if (
        summary[
            "output_size"
        ]
        != OUTPUT_SIZE
    ):
        raise AssertionError(
            "Clean-geometry output size mismatch."
        )

    if not math.isclose(
        float(
            summary[
                "alignment_scale"
            ]
        ),
        float(
            ALIGNMENT_SCALE
        ),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise AssertionError(
            "Clean-geometry alignment scale mismatch."
        )

    if (
        summary[
            "detector_upsample"
        ]
        != DETECTOR_UPSAMPLE
    ):
        raise AssertionError(
            "Clean-geometry detector upsample mismatch."
        )

    if (
        summary[
            "landmark_indices"
        ]
        != list(
            LANDMARK_INDICES
        )
    ):
        raise AssertionError(
            "Clean-geometry landmark indices mismatch."
        )

    if (
        summary[
            "predictor_sha256"
        ]
        != current_predictor_hash
    ):
        raise AssertionError(
            "Clean-geometry predictor hash mismatch."
        )

    return current_predictor_hash


def validate_temporal_membership(
    temporal_rows,
    geometry_rows,
):
    if (
        len(
            temporal_rows
        )
        != EXPECTED_BASE_VIDEOS
    ):
        raise AssertionError(
            "Temporal-plan video count mismatch: "
            "expected {}, got {}.".format(
                EXPECTED_BASE_VIDEOS,
                len(
                    temporal_rows
                ),
            )
        )

    if (
        len(
            geometry_rows
        )
        != EXPECTED_RECORDS
    ):
        raise AssertionError(
            "Clean-geometry record count mismatch: "
            "expected {}, got {}.".format(
                EXPECTED_RECORDS,
                len(
                    geometry_rows
                ),
            )
        )

    temporal_index = {}

    for row in temporal_rows:
        key = video_key(
            row
        )

        if key in temporal_index:
            raise AssertionError(
                "Duplicate temporal-plan video key: {}".format(
                    key
                )
            )

        if (
            row[
                "temporal_plan_status"
            ]
            != "ok"
        ):
            raise AssertionError(
                "Geometry input contains invalid "
                "temporal plan: {}".format(
                    key
                )
            )

        target_indices = (
            parse_json_cell(
                row[
                    "target_indices"
                ],
                description=(
                    "temporal-plan target_indices"
                ),
            )
        )

        if (
            len(
                target_indices
            )
            != TARGET_FRAME_BUDGET
        ):
            raise AssertionError(
                "Temporal-plan target count mismatch "
                "for {}".format(
                    key
                )
            )

        temporal_index[
            key
        ] = (
            row,
            target_indices,
        )

    geometry_by_video = defaultdict(
        list
    )

    frame_keys = set()

    for row in geometry_rows:
        key = video_key(
            row
        )

        if key not in temporal_index:
            raise AssertionError(
                "Clean-geometry row has no temporal-plan "
                "video: {}".format(
                    key
                )
            )

        current_frame_key = (
            frame_key(
                row
            )
        )

        if current_frame_key in frame_keys:
            raise AssertionError(
                "Duplicate clean-geometry frame key: {}".format(
                    current_frame_key
                )
            )

        frame_keys.add(
            current_frame_key
        )

        geometry_by_video[
            key
        ].append(
            row
        )

    if (
        set(
            geometry_by_video
        )
        != set(
            temporal_index
        )
    ):
        missing = sorted(
            set(
                temporal_index
            )
            - set(
                geometry_by_video
            )
        )

        extra = sorted(
            set(
                geometry_by_video
            )
            - set(
                temporal_index
            )
        )

        raise AssertionError(
            "Clean-geometry video membership mismatch.\n"
            "Missing: {}\n"
            "Extra: {}".format(
                missing[:5],
                extra[:5],
            )
        )

    return (
        temporal_index,
        geometry_by_video,
    )


def validate_video_records(
    key,
    temporal_row,
    target_indices,
    geometry_rows,
):
    if (
        len(
            geometry_rows
        )
        != TARGET_FRAME_BUDGET
    ):
        raise AssertionError(
            "Video {} has {} geometry rows instead of {}.".format(
                key,
                len(
                    geometry_rows
                ),
                TARGET_FRAME_BUDGET,
            )
        )

    geometry_rows = sorted(
        geometry_rows,
        key=lambda row: int(
            row[
                "temporal_position"
            ]
        ),
    )

    positions = [
        int(
            row[
                "temporal_position"
            ]
        )
        for row in geometry_rows
    ]

    expected_positions = list(
        range(
            TARGET_FRAME_BUDGET
        )
    )

    if (
        positions
        != expected_positions
    ):
        raise AssertionError(
            "Temporal positions mismatch for {}.\n"
            "Expected: {}\n"
            "Actual: {}".format(
                key,
                expected_positions,
                positions,
            )
        )

    valid_count = 0

    for row in geometry_rows:
        position = int(
            row[
                "temporal_position"
            ]
        )

        expected_frame_index = (
            target_indices[
                position
            ]
        )

        actual_frame_index = int(
            row[
                "source_frame_index"
            ]
        )

        if (
            actual_frame_index
            != expected_frame_index
        ):
            raise AssertionError(
                "Frozen frame-index mismatch for {} "
                "position {}: expected {}, got {}.".format(
                    key,
                    position,
                    expected_frame_index,
                    actual_frame_index,
                )
            )

        for field in (
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
                    field
                ]
                != temporal_row[
                    field
                ]
            ):
                raise AssertionError(
                    "Temporal-plan/geometry mismatch "
                    "for {} position {} field {}.".format(
                        key,
                        position,
                        field,
                    )
                )

        if (
            row[
                "alignment_method"
            ]
            != ALIGNMENT_METHOD
        ):
            raise AssertionError(
                "Alignment-method mismatch for {} "
                "position {}.".format(
                    key,
                    position,
                )
            )

        if not is_sha256(
            row[
                "source_frame_sha256"
            ]
        ):
            raise AssertionError(
                "Invalid source-frame SHA-256 for {} "
                "position {}.".format(
                    key,
                    position,
                )
            )

        status = row[
            "geometry_status"
        ]

        if (
            status
            not in (
                "valid",
                "invalid",
            )
        ):
            raise AssertionError(
                "Unexpected geometry status for {} "
                "position {}: {!r}".format(
                    key,
                    position,
                    status,
                )
            )

        if (
            status
            == "invalid"
        ):
            if not row[
                "failure_stage"
            ].strip():
                raise AssertionError(
                    "Invalid geometry row has no "
                    "failure_stage: {} position {}".format(
                        key,
                        position,
                    )
                )

            if not row[
                "failure_reason"
            ].strip():
                raise AssertionError(
                    "Invalid geometry row has no "
                    "failure_reason: {} position {}".format(
                        key,
                        position,
                    )
                )

            if any(
                row[
                    field
                ].strip()
                for field in (
                    "aligned_height",
                    "aligned_width",
                    "aligned_channels",
                    "aligned_sha256",
                )
            ):
                raise AssertionError(
                    "Invalid geometry row contains aligned "
                    "output metadata: {} position {}".format(
                        key,
                        position,
                    )
                )

            continue

        valid_count += 1

        if row[
            "failure_stage"
        ].strip():
            raise AssertionError(
                "Valid geometry row contains failure_stage: "
                "{} position {}".format(
                    key,
                    position,
                )
            )

        if row[
            "failure_reason"
        ].strip():
            raise AssertionError(
                "Valid geometry row contains failure_reason: "
                "{} position {}".format(
                    key,
                    position,
                )
            )

        face_count = int(
            row[
                "face_count"
            ]
        )

        if face_count < 1:
            raise AssertionError(
                "Valid geometry has face_count < 1: "
                "{} position {}".format(
                    key,
                    position,
                )
            )

        bbox = parse_json_cell(
            row[
                "bbox_json"
            ],
            description="bbox_json",
        )

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
                isinstance(
                    value,
                    int,
                )
                for value in bbox
            )
        ):
            raise AssertionError(
                "Invalid bbox for {} position {}: {}".format(
                    key,
                    position,
                    bbox,
                )
            )

        if (
            bbox[2]
            <= bbox[0]
            or bbox[3]
            <= bbox[1]
        ):
            raise AssertionError(
                "Non-positive bbox area for {} "
                "position {}: {}".format(
                    key,
                    position,
                    bbox,
                )
            )

        keypoints = np.asarray(
            parse_json_cell(
                row[
                    "keypoints_json"
                ],
                description="keypoints_json",
            ),
            dtype=np.float64,
        )

        if (
            keypoints.shape
            != (
                5,
                2,
            )
        ):
            raise AssertionError(
                "Invalid keypoint shape for {} "
                "position {}: {}".format(
                    key,
                    position,
                    keypoints.shape,
                )
            )

        if not np.isfinite(
            keypoints
        ).all():
            raise AssertionError(
                "Non-finite keypoints for {} "
                "position {}.".format(
                    key,
                    position,
                )
            )

        affine = np.asarray(
            parse_json_cell(
                row[
                    "affine_matrix_json"
                ],
                description="affine_matrix_json",
            ),
            dtype=np.float64,
        )

        if (
            affine.shape
            != (
                2,
                3,
            )
        ):
            raise AssertionError(
                "Invalid affine-matrix shape for {} "
                "position {}: {}".format(
                    key,
                    position,
                    affine.shape,
                )
            )

        if not np.isfinite(
            affine
        ).all():
            raise AssertionError(
                "Non-finite affine matrix for {} "
                "position {}.".format(
                    key,
                    position,
                )
            )

        reconstructed = (
            estimate_similarity_matrix(
                keypoints=(
                    keypoints.astype(
                        np.float32
                    )
                ),
                output_size=(
                    OUTPUT_SIZE
                ),
                scale=(
                    ALIGNMENT_SCALE
                ),
            )
        )

        if reconstructed is None:
            raise AssertionError(
                "Saved valid keypoints no longer produce "
                "a similarity matrix for {} position {}.".format(
                    key,
                    position,
                )
            )

        if not np.allclose(
            affine,
            reconstructed,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise AssertionError(
                "Saved affine matrix differs from "
                "matrix reconstructed from saved "
                "keypoints for {} position {}.\n"
                "Saved:\n{}\n"
                "Reconstructed:\n{}".format(
                    key,
                    position,
                    affine,
                    reconstructed,
                )
            )

        if (
            int(
                row[
                    "aligned_height"
                ]
            )
            != OUTPUT_SIZE
            or int(
                row[
                    "aligned_width"
                ]
            )
            != OUTPUT_SIZE
            or int(
                row[
                    "aligned_channels"
                ]
            )
            != 3
        ):
            raise AssertionError(
                "Aligned-image geometry mismatch for {} "
                "position {}.".format(
                    key,
                    position,
                )
            )

        if not is_sha256(
            row[
                "aligned_sha256"
            ]
        ):
            raise AssertionError(
                "Invalid aligned SHA-256 for {} "
                "position {}.".format(
                    key,
                    position,
                )
            )

    return valid_count


def validate_summary(
    geometry_rows,
    valid_counts_by_video,
):
    summary = read_json(
        SUMMARY_PATH
    )

    predictor_hash = (
        validate_training_contract(
            summary
        )
    )

    actual_geometry_hash = (
        sha256_file(
            CLEAN_GEOMETRY_PATH
        )
    )

    if (
        summary[
            "clean_geometry_sha256"
        ]
        != actual_geometry_hash
    ):
        raise AssertionError(
            "clean_geometry.csv SHA-256 mismatch."
        )

    actual_temporal_hash = (
        sha256_file(
            TEMPORAL_PLAN_PATH
        )
    )

    if (
        summary[
            "temporal_plan_sha256"
        ]
        != actual_temporal_hash
    ):
        raise AssertionError(
            "Temporal-plan SHA-256 mismatch."
        )

    if (
        summary[
            "selected_video_count"
        ]
        != EXPECTED_BASE_VIDEOS
    ):
        raise AssertionError(
            "Summary selected_video_count mismatch."
        )

    if (
        summary[
            "targets_per_video"
        ]
        != TARGET_FRAME_BUDGET
    ):
        raise AssertionError(
            "Summary targets_per_video mismatch."
        )

    if (
        summary[
            "nominal_target_count"
        ]
        != EXPECTED_RECORDS
    ):
        raise AssertionError(
            "Summary nominal_target_count mismatch."
        )

    actual_status_counts = dict(
        sorted(
            Counter(
                row[
                    "geometry_status"
                ]
                for row in geometry_rows
            ).items()
        )
    )

    if (
        summary[
            "geometry_status_counts"
        ]
        != actual_status_counts
    ):
        raise AssertionError(
            "Summary geometry_status_counts mismatch."
        )

    actual_failure_stages = dict(
        sorted(
            Counter(
                row[
                    "failure_stage"
                ]
                for row in geometry_rows
                if (
                    row[
                        "geometry_status"
                    ]
                    == "invalid"
                )
            ).items()
        )
    )

    if (
        summary[
            "failure_stages"
        ]
        != actual_failure_stages
    ):
        raise AssertionError(
            "Summary failure_stages mismatch."
        )

    distribution = Counter(
        valid_counts_by_video.values()
    )

    actual_distribution = {
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

    if (
        summary[
            "clean_valid_frame_count_distribution"
        ]
        != actual_distribution
    ):
        raise AssertionError(
            "Summary clean-valid distribution mismatch."
        )

    counts = list(
        valid_counts_by_video.values()
    )

    if (
        summary[
            "minimum_clean_valid_frames"
        ]
        != min(
            counts
        )
    ):
        raise AssertionError(
            "Summary minimum_clean_valid_frames mismatch."
        )

    if (
        summary[
            "maximum_clean_valid_frames"
        ]
        != max(
            counts
        )
    ):
        raise AssertionError(
            "Summary maximum_clean_valid_frames mismatch."
        )

    if (
        summary[
            "complete_32_video_count"
        ]
        != distribution.get(
            TARGET_FRAME_BUDGET,
            0,
        )
    ):
        raise AssertionError(
            "Summary complete_32_video_count mismatch."
        )

    if (
        summary[
            "zero_valid_video_count"
        ]
        != distribution.get(
            0,
            0,
        )
    ):
        raise AssertionError(
            "Summary zero_valid_video_count mismatch."
        )

    return (
        summary,
        predictor_hash,
        actual_geometry_hash,
    )


def main():
    temporal_rows = read_csv(
        TEMPORAL_PLAN_PATH
    )

    geometry_rows = read_csv(
        CLEAN_GEOMETRY_PATH
    )

    (
        temporal_index,
        geometry_by_video,
    ) = validate_temporal_membership(
        temporal_rows=temporal_rows,
        geometry_rows=geometry_rows,
    )

    valid_counts_by_video = {}

    for key in sorted(
        temporal_index
    ):
        (
            temporal_row,
            target_indices,
        ) = temporal_index[
            key
        ]

        valid_count = (
            validate_video_records(
                key=key,
                temporal_row=temporal_row,
                target_indices=target_indices,
                geometry_rows=(
                    geometry_by_video[
                        key
                    ]
                ),
            )
        )

        valid_counts_by_video[
            key
        ] = valid_count

    (
        summary,
        predictor_hash,
        geometry_hash,
    ) = validate_summary(
        geometry_rows=geometry_rows,
        valid_counts_by_video=(
            valid_counts_by_video
        ),
    )

    valid_records = sum(
        count
        for count in (
            valid_counts_by_video.values()
        )
    )

    invalid_records = (
        EXPECTED_RECORDS
        - valid_records
    )

    complete_32 = sum(
        count
        == TARGET_FRAME_BUDGET
        for count in (
            valid_counts_by_video.values()
        )
    )

    incomplete = (
        EXPECTED_BASE_VIDEOS
        - complete_32
    )

    print(
        "CLEAN-REFERENCE GEOMETRY VALIDATION"
    )
    print(
        "  base videos:               {}".format(
            EXPECTED_BASE_VIDEOS
        )
    )
    print(
        "  nominal positions:         {}".format(
            EXPECTED_RECORDS
        )
    )
    print(
        "  clean-valid positions:     {}".format(
            valid_records
        )
    )
    print(
        "  clean-invalid positions:   {}".format(
            invalid_records
        )
    )
    print(
        "  complete-32 videos:        {}".format(
            complete_32
        )
    )
    print(
        "  incomplete videos:         {}".format(
            incomplete
        )
    )
    print(
        "  zero-valid videos:         {}".format(
            summary[
                "zero_valid_video_count"
            ]
        )
    )
    print(
        "  failure stages:            {}".format(
            summary[
                "failure_stages"
            ]
        )
    )
    print(
        "  valid-count distribution:  {}".format(
            summary[
                "clean_valid_frame_count_distribution"
            ]
        )
    )
    print(
        "  alignment method:          {}".format(
            ALIGNMENT_METHOD
        )
    )
    print(
        "  predictor SHA-256:         {}".format(
            predictor_hash
        )
    )
    print(
        "  geometry SHA-256:          {}".format(
            geometry_hash
        )
    )
    print()
    print(
        "CLEAN-REFERENCE GEOMETRY VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()