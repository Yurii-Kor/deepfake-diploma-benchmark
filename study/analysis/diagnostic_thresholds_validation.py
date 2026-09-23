from __future__ import annotations

import copy
import math

from study.analysis.diagnostic_thresholds import (
    compute_diagnostic_thresholds,
)


CHECKPOINT = (
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)

PRIMARY_THRESHOLD = 0.20

PRIMARY_THRESHOLD_ID = (
    "xception_ffpp_validation_cln_fpr005_aaaaaaaaaaaa"
)


def assert_close(
    actual,
    expected,
    *,
    tolerance=1e-15,
):
    assert math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=tolerance,
    ), (
        "Expected {}, got {}."
        .format(
            expected,
            actual,
        )
    )


def assert_raises(
    expected_exception,
    function,
):
    passed = False

    try:
        function()

    except expected_exception:
        passed = True

    assert passed


def primary_artifact():
    return {
        "schema_version": 1,

        "artifact_type": (
            "primary_operating_thresholds"
        ),

        "target_fpr": 0.05,

        "threshold_count": 1,

        "thresholds": [
            {
                "schema_version": 1,

                "threshold_id": (
                    PRIMARY_THRESHOLD_ID
                ),

                "detector": "xception",

                "checkpoint_sha256": (
                    CHECKPOINT
                ),

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

                "threshold_value": (
                    PRIMARY_THRESHOLD
                ),

                "target_fpr": 0.05,
            }
        ],
    }


def record(
    *,
    video_id,
    condition,
    score,
    eligible=True,
    label=0,
    dataset="FaceForensics++",
    role="validation",
):
    path = (
        "original/{:03d}.mp4".format(
            video_id
        )
    )

    return {
        "schema_version": 1,

        "detector": "xception",

        "checkpoint_sha256": (
            CHECKPOINT
        ),

        "dataset": (
            dataset
        ),

        "role": (
            role
        ),

        "pairing_uid": (
            "{}::{}".format(
                dataset,
                path,
            )
        ),

        "relative_source_path": (
            path
        ),

        "study_label": (
            label
        ),

        "condition": (
            condition
        ),

        "analysis_eligible": (
            eligible
        ),

        "analysis_status": (
            "valid_score"
            if eligible
            else "invalid_inference"
        ),

        "video_score": (
            score
            if eligible
            else None
        ),
    }


def find_result(
    results,
    *,
    condition,
):
    matches = [
        row
        for row in results
        if (
            row[
                "condition"
            ]
            == condition
        )
    ]

    assert len(
        matches
    ) == 1

    return matches[
        0
    ]


def main():
    rows = []

    #
    # --------------------------------------------------
    # RSZ:
    #
    # 20 real scores:
    #   .11, .12, ..., .30
    #
    # At alpha=.05 and n=20 exactly one FP is allowed.
    #
    # threshold=.30 -> 1/20 = .05
    # threshold=.29 -> 2/20 = .10
    #
    # Therefore:
    #
    #   t_RSZ = .30
    #   t_CLN = .20
    #   delta = +.10
    # --------------------------------------------------
    #
    for index in range(
        1,
        21,
    ):
        rows.append(
            record(
                video_id=index,
                condition="RSZ",
                score=(
                    0.10
                    + index / 100.0
                ),
            )
        )

    #
    # BLR:
    #
    # .01, .02, ..., .20
    #
    # Diagnostic threshold exactly equals the frozen
    # primary clean threshold:
    #
    #   t_BLR = .20
    #   delta = 0
    # --------------------------------------------------
    #
    for index in range(
        1,
        21,
    ):
        rows.append(
            record(
                video_id=index,
                condition="BLR",
                score=(
                    index / 100.0
                ),
            )
        )

    #
    # H40:
    #
    # .005, .010, ..., .100
    #
    # threshold=.100 produces one FP.
    #
    #   t_H40 = .10
    #   delta = -.10
    # --------------------------------------------------
    #
    for index in range(
        1,
        21,
    ):
        rows.append(
            record(
                video_id=index,
                condition="H40",
                score=(
                    index / 200.0
                ),
            )
        )

    #
    # Add irrelevant records deliberately.
    #
    # CLN must not be re-estimated by this function.
    #
    for index in range(
        1,
        21,
    ):
        rows.append(
            record(
                video_id=index,
                condition="CLN",
                score=0.99,
            )
        )

    #
    # Manipulated validation records must not
    # participate.
    #
    rows.append(
        record(
            video_id=900,
            condition="RSZ",
            score=1.0,
            label=1,
        )
    )

    #
    # FF++ test must not participate.
    #
    rows.append(
        record(
            video_id=901,
            condition="RSZ",
            score=1.0,
            role="test",
        )
    )

    #
    # Celeb must not participate.
    #
    rows.append(
        record(
            video_id=902,
            condition="RSZ",
            score=1.0,
            dataset="Celeb-DF-v2",
            role="external_evaluation",
        )
    )

    results = (
        compute_diagnostic_thresholds(
            analysis_records=rows,
            primary_threshold_artifact=(
                primary_artifact()
            ),
        )
    )

    assert len(
        results
    ) == 3

    #
    # --------------------------------------------------
    # 1. Positive displacement.
    # --------------------------------------------------
    #
    rsz = find_result(
        results,
        condition="RSZ",
    )

    assert (
        rsz[
            "valid_real_records"
        ]
        == 20
    )

    assert (
        rsz[
            "diagnostic_false_positives"
        ]
        == 1
    )

    assert_close(
        rsz[
            "diagnostic_realized_fpr"
        ],
        0.05,
    )

    assert_close(
        rsz[
            "primary_clean_threshold"
        ],
        0.20,
    )

    assert_close(
        rsz[
            "diagnostic_threshold"
        ],
        0.30,
    )

    assert_close(
        rsz[
            "threshold_displacement"
        ],
        0.10,
    )

    #
    # --------------------------------------------------
    # 2. Zero displacement.
    # --------------------------------------------------
    #
    blr = find_result(
        results,
        condition="BLR",
    )

    assert_close(
        blr[
            "diagnostic_threshold"
        ],
        0.20,
    )

    assert_close(
        blr[
            "threshold_displacement"
        ],
        0.0,
    )

    #
    # --------------------------------------------------
    # 3. Negative displacement.
    # --------------------------------------------------
    #
    h40 = find_result(
        results,
        condition="H40",
    )

    assert_close(
        h40[
            "diagnostic_threshold"
        ],
        0.10,
    )

    assert_close(
        h40[
            "threshold_displacement"
        ],
        -0.10,
    )

    #
    # --------------------------------------------------
    # 4. Processed-condition failure changes the
    #    diagnostic sample but MUST NOT alter the
    #    already frozen primary clean threshold.
    #
    # Remove the RSZ video with score .30.
    #
    # n becomes 19.
    #
    # floor(.05 * 19) = 0, so the diagnostic threshold
    # becomes the representable value immediately above
    # the new maximum .29.
    #
    # The clean reference remains exactly .20.
    # --------------------------------------------------
    #
    failure_rows = copy.deepcopy(
        rows
    )

    failed = next(
        row
        for row in failure_rows
        if (
            row[
                "condition"
            ]
            == "RSZ"
            and row[
                "relative_source_path"
            ]
            == "original/020.mp4"
        )
    )

    failed[
        "analysis_eligible"
    ] = False

    failed[
        "analysis_status"
    ] = "invalid_inference"

    failed[
        "video_score"
    ] = None

    failure_results = (
        compute_diagnostic_thresholds(
            analysis_records=(
                failure_rows
            ),
            primary_threshold_artifact=(
                primary_artifact()
            ),
        )
    )

    failure_rsz = find_result(
        failure_results,
        condition="RSZ",
    )

    assert (
        failure_rsz[
            "nominal_real_records"
        ]
        == 20
    )

    assert (
        failure_rsz[
            "valid_real_records"
        ]
        == 19
    )

    assert (
        failure_rsz[
            "excluded_invalid_real_records"
        ]
        == 1
    )

    assert_close(
        failure_rsz[
            "primary_clean_threshold"
        ],
        0.20,
    )

    assert (
        failure_rsz[
            "diagnostic_threshold"
        ]
        > 0.29
    )

    assert (
        failure_rsz[
            "diagnostic_threshold"
        ]
        < 0.30
    )

    assert (
        failure_rsz[
            "clean_threshold_reestimated"
        ]
        is False
    )

    assert (
        failure_rsz[
            "processed_failure_changes_primary_threshold"
        ]
        is False
    )

    #
    # --------------------------------------------------
    # 5. Duplicate validation identity is forbidden.
    # --------------------------------------------------
    #
    duplicate_rows = list(
        rows
    )

    duplicate_rows.append(
        copy.deepcopy(
            rows[
                0
            ]
        )
    )

    assert_raises(
        ValueError,
        lambda: (
            compute_diagnostic_thresholds(
                analysis_records=(
                    duplicate_rows
                ),
                primary_threshold_artifact=(
                    primary_artifact()
                ),
            )
        ),
    )

    #
    # --------------------------------------------------
    # 6. Checkpoint mismatch is forbidden.
    # --------------------------------------------------
    #
    bad_checkpoint_artifact = (
        primary_artifact()
    )

    bad_checkpoint_artifact[
        "thresholds"
    ][
        0
    ][
        "checkpoint_sha256"
    ] = (
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    )

    assert_raises(
        ValueError,
        lambda: (
            compute_diagnostic_thresholds(
                analysis_records=rows,
                primary_threshold_artifact=(
                    bad_checkpoint_artifact
                ),
            )
        ),
    )

    #
    # --------------------------------------------------
    # 7. Wrong target FPR artifact rejected.
    # --------------------------------------------------
    #
    bad_target_artifact = (
        primary_artifact()
    )

    bad_target_artifact[
        "target_fpr"
    ] = 0.10

    assert_raises(
        ValueError,
        lambda: (
            compute_diagnostic_thresholds(
                analysis_records=rows,
                primary_threshold_artifact=(
                    bad_target_artifact
                ),
            )
        ),
    )

    print(
        "DIAGNOSTIC THRESHOLD CONTRACT VALIDATION"
    )
    print(
        "  FF++ validation real only:     PASSED"
    )
    print(
        "  degraded conditions only:      PASSED"
    )
    print(
        "  same target FPR = .05:         PASSED"
    )
    print(
        "  positive displacement:         PASSED"
    )
    print(
        "  zero displacement:             PASSED"
    )
    print(
        "  negative displacement:         PASSED"
    )
    print(
        "  CLN not re-estimated here:     PASSED"
    )
    print(
        "  processed failure accounting:  PASSED"
    )
    print(
        "  frozen clean threshold stable: PASSED"
    )
    print(
        "  duplicate rejection:           PASSED"
    )
    print(
        "  checkpoint consistency:        PASSED"
    )
    print(
        "  target-FPR consistency:        PASSED"
    )
    print()
    print(
        "DIAGNOSTIC THRESHOLD CONTRACT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()