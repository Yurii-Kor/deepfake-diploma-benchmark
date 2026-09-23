from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from study.evaluation.inference_validity import (
    FAILURE_CONDITION_INTEGRITY,
    STATUS_INSUFFICIENT_VALID_INPUT,
    STATUS_INVALID_CONDITION,
    STATUS_OK,
    finalize_video_inference,
    paired_valid_records,
)


STUDY_ROOT = (
    Path.home()
    / "deepfake_lab"
    / "study_data"
)

CLEAN_GEOMETRY_PATH = (
    STUDY_ROOT
    / "evaluation"
    / "clean_geometry.csv"
)

FRAME_SCORING_PATH = (
    STUDY_ROOT
    / "evaluation"
    / "frame_scoring_smoke.jsonl"
)


def read_csv(
    path,
):
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


def read_jsonl(
    path,
):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                rows.append(
                    json.loads(
                        line
                    )
                )

    return rows


def identity_from_row(
    row,
):
    return {
        "dataset": (
            row[
                "dataset"
            ]
        ),
        "role": (
            row[
                "role"
            ]
        ),
        "subgroup": (
            row[
                "subgroup"
            ]
        ),
        "study_label": int(
            row[
                "study_label"
            ]
        ),
        "source_label": (
            row[
                "source_label"
            ]
        ),
        "base_video_id": (
            row[
                "base_video_id"
            ]
        ),
        "relative_source_path": (
            row[
                "relative_source_path"
            ]
        ),
    }


def main():
    geometry_rows = read_csv(
        CLEAN_GEOMETRY_PATH
    )

    frame_rows = read_jsonl(
        FRAME_SCORING_PATH
    )

    geometry_by_video = defaultdict(
        list
    )

    for row in geometry_rows:
        key = (
            row[
                "dataset"
            ],
            row[
                "role"
            ],
            row[
                "relative_source_path"
            ],
        )

        geometry_by_video[
            key
        ].append(
            row
        )

    #
    # ------------------------------------------------------
    # 1. REAL 32/32 smoke group must remain valid.
    # ------------------------------------------------------
    #
    real_group = [
        row
        for row in frame_rows
        if (
            row[
                "detector"
            ]
            == "xception"
            and row[
                "condition"
            ]
            == "CLN"
            and row[
                "base_video_id"
            ]
            == "004"
        )
    ]

    if (
        len(
            real_group
        )
        != 32
    ):
        raise RuntimeError(
            "Expected 32 real smoke frame records."
        )

    real_positions = sorted(
        int(
            row[
                "temporal_position"
            ]
        )
        for row in real_group
    )

    valid_complete = (
        finalize_video_inference(
            identity=identity_from_row(
                real_group[
                    0
                ]
            ),
            condition="CLN",
            detector="xception",
            clean_valid_positions=(
                real_positions
            ),
            frame_records=(
                real_group
            ),
        )
    )

    assert (
        valid_complete[
            "inference_status"
        ]
        == STATUS_OK
    )

    assert (
        valid_complete[
            "clean_valid_frame_count"
        ]
        == 32
    )

    assert (
        valid_complete[
            "successful_frame_count"
        ]
        == 32
    )

    expected_mean = (
        sum(
            float(
                row[
                    "score"
                ]
            )
            for row in real_group
        )
        / 32
    )

    assert (
        valid_complete[
            "video_score"
        ]
        == expected_mean
    )

    #
    # ------------------------------------------------------
    # 2. Find the REAL minimum-valid clean geometry video.
    #    Current corpus should contain the 19/32 case.
    # ------------------------------------------------------
    #
    valid_counts = {}

    for key, rows in (
        geometry_by_video.items()
    ):
        count = sum(
            row[
                "geometry_status"
            ]
            == "valid"
            for row in rows
        )

        valid_counts[
            key
        ] = count

    incomplete_key = min(
        valid_counts,
        key=lambda key: (
            valid_counts[
                key
            ],
            key,
        ),
    )

    incomplete_count = (
        valid_counts[
            incomplete_key
        ]
    )

    if (
        incomplete_count
        != 19
    ):
        raise RuntimeError(
            "Expected the frozen corpus minimum "
            "clean-valid count to be 19, got {} "
            "for {}.".format(
                incomplete_count,
                incomplete_key,
            )
        )

    incomplete_geometry = sorted(
        geometry_by_video[
            incomplete_key
        ],
        key=lambda row: int(
            row[
                "temporal_position"
            ]
        ),
    )

    incomplete_valid_rows = [
        row
        for row in (
            incomplete_geometry
        )
        if (
            row[
                "geometry_status"
            ]
            == "valid"
        )
    ]

    incomplete_positions = [
        int(
            row[
                "temporal_position"
            ]
        )
        for row in (
            incomplete_valid_rows
        )
    ]

    #
    # Synthetic detector outputs, but REAL frozen 19-position set.
    #
    synthetic_success = [
        {
            "temporal_position": (
                position
            ),
            "score": (
                0.1
                + index
                * 0.01
            ),
            "inference_status": (
                STATUS_OK
            ),
            "failure_stage": "",
            "failure_reason": "",
        }
        for index, position in enumerate(
            incomplete_positions
        )
    ]

    valid_incomplete = (
        finalize_video_inference(
            identity=identity_from_row(
                incomplete_valid_rows[
                    0
                ]
            ),
            condition="CLN",
            detector="xception",
            clean_valid_positions=(
                incomplete_positions
            ),
            frame_records=(
                synthetic_success
            ),
        )
    )

    assert (
        valid_incomplete[
            "inference_status"
        ]
        == STATUS_OK
    )

    assert (
        valid_incomplete[
            "clean_valid_frame_count"
        ]
        == 19
    )

    assert (
        valid_incomplete[
            "successful_frame_count"
        ]
        == 19
    )

    assert (
        valid_incomplete[
            "video_score"
        ]
        is not None
    )

    #
    # ------------------------------------------------------
    # 3. Zero clean-valid positions -> insufficient input.
    # ------------------------------------------------------
    #
    zero_valid = (
        finalize_video_inference(
            identity=identity_from_row(
                incomplete_valid_rows[
                    0
                ]
            ),
            condition="CLN",
            detector="xception",
            clean_valid_positions=[],
            frame_records=[],
        )
    )

    assert (
        zero_valid[
            "inference_status"
        ]
        == STATUS_INSUFFICIENT_VALID_INPUT
    )

    assert (
        zero_valid[
            "video_score"
        ]
        is None
    )

    #
    # ------------------------------------------------------
    # 4. Missing one required position -> invalid, no mean.
    # ------------------------------------------------------
    #
    missing_one = (
        synthetic_success[
            :-1
        ]
    )

    invalid_missing = (
        finalize_video_inference(
            identity=identity_from_row(
                incomplete_valid_rows[
                    0
                ]
            ),
            condition="RSZ",
            detector="xception",
            clean_valid_positions=(
                incomplete_positions
            ),
            frame_records=(
                missing_one
            ),
        )
    )

    assert (
        invalid_missing[
            "inference_status"
        ]
        == STATUS_INVALID_CONDITION
    )

    assert (
        invalid_missing[
            "video_score"
        ]
        is None
    )

    #
    # ------------------------------------------------------
    # 5. Substitute position does NOT repair a missing one.
    # ------------------------------------------------------
    #
    substitute_position = next(
        position
        for position in range(
            32
        )
        if (
            position
            not in incomplete_positions
        )
    )

    substituted = [
        dict(
            row
        )
        for row in missing_one
    ]

    substituted.append(
        {
            "temporal_position": (
                substitute_position
            ),
            "score": 0.5,
            "inference_status": (
                STATUS_OK
            ),
            "failure_stage": "",
            "failure_reason": "",
        }
    )

    invalid_substitution = (
        finalize_video_inference(
            identity=identity_from_row(
                incomplete_valid_rows[
                    0
                ]
            ),
            condition="RSZ",
            detector="xception",
            clean_valid_positions=(
                incomplete_positions
            ),
            frame_records=(
                substituted
            ),
        )
    )

    assert (
        invalid_substitution[
            "inference_status"
        ]
        == STATUS_INVALID_CONDITION
    )

    assert (
        invalid_substitution[
            "video_score"
        ]
        is None
    )

    #
    # ------------------------------------------------------
    # 6. Explicit frame inference failure invalidates video.
    # ------------------------------------------------------
    #
    failed_rows = [
        dict(
            row
        )
        for row in synthetic_success
    ]

    failed_rows[
        3
    ][
        "inference_status"
    ] = STATUS_INVALID_CONDITION

    failed_rows[
        3
    ][
        "failure_stage"
    ] = "detector_inference"

    failed_rows[
        3
    ][
        "failure_reason"
    ] = "synthetic contract-test failure"

    invalid_frame_failure = (
        finalize_video_inference(
            identity=identity_from_row(
                incomplete_valid_rows[
                    0
                ]
            ),
            condition="RSZ",
            detector="xception",
            clean_valid_positions=(
                incomplete_positions
            ),
            frame_records=(
                failed_rows
            ),
        )
    )

    assert (
        invalid_frame_failure[
            "inference_status"
        ]
        == STATUS_INVALID_CONDITION
    )

    assert (
        invalid_frame_failure[
            "video_score"
        ]
        is None
    )

    #
    # ------------------------------------------------------
    # 7. Non-finite required score invalidates the video.
    # ------------------------------------------------------
    #
    nonfinite_rows = [
        dict(
            row
        )
        for row in synthetic_success
    ]

    nonfinite_rows[
        4
    ][
        "score"
    ] = float(
        "nan"
    )

    invalid_nonfinite = (
        finalize_video_inference(
            identity=identity_from_row(
                incomplete_valid_rows[
                    0
                ]
            ),
            condition="RSZ",
            detector="xception",
            clean_valid_positions=(
                incomplete_positions
            ),
            frame_records=(
                nonfinite_rows
            ),
        )
    )

    assert (
        invalid_nonfinite[
            "inference_status"
        ]
        == STATUS_INVALID_CONDITION
    )

    assert (
        invalid_nonfinite[
            "video_score"
        ]
        is None
    )

    #
    # ------------------------------------------------------
    # 8. Upstream condition-integrity failure -> no score.
    # ------------------------------------------------------
    #
    invalid_integrity = (
        finalize_video_inference(
            identity=identity_from_row(
                incomplete_valid_rows[
                    0
                ]
            ),
            condition="RSZ",
            detector="xception",
            clean_valid_positions=(
                incomplete_positions
            ),
            frame_records=[],
            upstream_failure_stage=(
                FAILURE_CONDITION_INTEGRITY
            ),
            upstream_failure_reason=(
                "synthetic contract-test "
                "processed-video integrity failure"
            ),
        )
    )

    assert (
        invalid_integrity[
            "inference_status"
        ]
        == STATUS_INVALID_CONDITION
    )

    assert (
        invalid_integrity[
            "video_score"
        ]
        is None
    )

    #
    # ------------------------------------------------------
    # 9. Paired comparison = intersection of valid records.
    # ------------------------------------------------------
    #
    clean_records = [
        valid_complete,
        valid_incomplete,
    ]

    processed_records = [
        {
            **valid_complete,
            "condition": "RSZ",
        },
        invalid_missing,
    ]

    paired = (
        paired_valid_records(
            clean_records=(
                clean_records
            ),
            processed_records=(
                processed_records
            ),
        )
    )

    assert (
        len(
            paired
        )
        == 1
    )

    assert (
        paired[
            0
        ][
            0
        ][
            "base_video_id"
        ]
        == "004"
    )

    print(
        "INFERENCE VALIDITY CONTRACT VALIDATION"
    )
    print(
        "  real complete-32 record:      PASSED"
    )
    print(
        "  frozen minimum-valid video:   {} | {}/32".format(
            incomplete_key,
            incomplete_count,
        )
    )
    print(
        "  >=1 clean-valid remains valid: PASSED"
    )
    print(
        "  zero clean-valid invalid:      PASSED"
    )
    print(
        "  missing required frame:        PASSED"
    )
    print(
        "  substitution forbidden:        PASSED"
    )
    print(
        "  frame inference failure:       PASSED"
    )
    print(
        "  non-finite score failure:      PASSED"
    )
    print(
        "  condition-integrity failure:   PASSED"
    )
    print(
        "  paired valid intersection:     PASSED"
    )
    print()
    print(
        "INFERENCE VALIDITY CONTRACT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()