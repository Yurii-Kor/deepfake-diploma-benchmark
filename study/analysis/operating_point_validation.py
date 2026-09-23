from __future__ import annotations

import copy
import math

from study.analysis.operating_point import (
    PRIMARY_TARGET_FPR,
    apply_frozen_threshold,
    decision_from_score,
    select_primary_low_fpr_threshold,
    select_primary_thresholds_from_analysis_records,
)


CHECKPOINT = (
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)


def analysis_row(
    *,
    video_number,
    score,
    condition="CLN",
    dataset="FaceForensics++",
    role="validation",
    study_label=0,
    detector="xception",
    eligible=True,
):
    path = (
        "original/{:03d}.mp4".format(
            video_number
        )
        if study_label == 0
        else (
            "Deepfakes/"
            "{:03d}_{:03d}.mp4"
            .format(
                video_number,
                (
                    video_number
                    + 1
                )
                % 1000,
            )
        )
    )

    return {
        "schema_version": 1,

        "detector": detector,
        "checkpoint_sha256": (
            CHECKPOINT
        ),

        "dataset": dataset,
        "role": role,

        "pairing_uid": (
            "{}::{}".format(
                dataset,
                path,
            )
        ),

        "study_label": (
            study_label
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


def main():
    #
    # --------------------------------------------------
    # 1. Exact 5% operating point with n=140.
    #
    # Scores:
    #
    # 0.001, 0.002, ..., 0.140
    #
    # At threshold 0.134 exactly seven scores are
    # >= threshold.
    #
    # The next lower candidate, 0.133, produces eight
    # false positives and therefore violates alpha=.05.
    # --------------------------------------------------
    #
    exact_scores = [
        index
        / 1000.0
        for index in range(
            1,
            141,
        )
    ]

    exact = (
        select_primary_low_fpr_threshold(
            real_scores=(
                exact_scores
            ),
            target_fpr=(
                PRIMARY_TARGET_FPR
            ),
        )
    )

    assert exact[
        "n_real"
    ] == 140

    assert exact[
        "max_allowed_false_positives"
    ] == 7

    assert exact[
        "false_positives"
    ] == 7

    assert math.isclose(
        exact[
            "threshold_value"
        ],
        0.134,
        rel_tol=0.0,
        abs_tol=1e-15,
    )

    assert math.isclose(
        exact[
            "realized_fpr"
        ],
        0.05,
        rel_tol=0.0,
        abs_tol=1e-15,
    )

    #
    # --------------------------------------------------
    # 2. Tied scores make exact 5% unattainable.
    #
    # 6 scores at .9
    # 3 scores at .8
    # 131 scores at .1
    #
    # .8 -> 9/140 > .05
    # .9 -> 6/140 <= .05
    #
    # Therefore .9 is the lowest admissible candidate.
    # --------------------------------------------------
    #
    tied_scores = (
        [0.9] * 6
        + [0.8] * 3
        + [0.1] * 131
    )

    tied = (
        select_primary_low_fpr_threshold(
            real_scores=(
                tied_scores
            ),
            target_fpr=0.05,
        )
    )

    assert tied[
        "false_positives"
    ] == 6

    assert math.isclose(
        tied[
            "threshold_value"
        ],
        0.9,
        rel_tol=0.0,
        abs_tol=1e-15,
    )

    assert math.isclose(
        tied[
            "realized_fpr"
        ],
        6 / 140,
        rel_tol=0.0,
        abs_tol=1e-15,
    )

    #
    # --------------------------------------------------
    # 3. Clean real validation is the ONLY primary
    #    threshold-selection source.
    # --------------------------------------------------
    #
    rows = [
        analysis_row(
            video_number=index,
            score=index / 1000.0,
        )
        for index in range(
            1,
            141,
        )
    ]

    #
    # Same validation videos under RSZ with extreme
    # scores. These MUST NOT alter the primary threshold.
    #
    rows.extend(
        analysis_row(
            video_number=index,
            score=1.0,
            condition="RSZ",
        )
        for index in range(
            1,
            141,
        )
    )

    #
    # Manipulated validation records are irrelevant
    # to the real-only primary operating point.
    #
    rows.extend(
        analysis_row(
            video_number=index,
            score=0.0,
            study_label=1,
        )
        for index in range(
            1,
            21,
        )
    )

    #
    # Held-out FF++ test data must not participate.
    #
    rows.extend(
        analysis_row(
            video_number=index,
            score=1.0,
            role="test",
        )
        for index in range(
            200,
            220,
        )
    )

    #
    # External-domain data must not participate.
    #
    rows.extend(
        analysis_row(
            video_number=index,
            score=1.0,
            dataset="Celeb-DF-v2",
            role="external_evaluation",
        )
        for index in range(
            300,
            320,
        )
    )

    thresholds = (
        select_primary_thresholds_from_analysis_records(
            rows,
            target_fpr=0.05,
        )
    )

    assert len(
        thresholds
    ) == 1

    primary = thresholds[
        0
    ]

    assert (
        primary[
            "calibration_dataset"
        ]
        == "FaceForensics++"
    )

    assert (
        primary[
            "calibration_role"
        ]
        == "validation"
    )

    assert (
        primary[
            "calibration_condition"
        ]
        == "CLN"
    )

    assert (
        primary[
            "calibration_class"
        ]
        == "real_only"
    )

    assert (
        primary[
            "nominal_real_records"
        ]
        == 140
    )

    assert (
        primary[
            "valid_real_records"
        ]
        == 140
    )

    assert (
        primary[
            "excluded_invalid_real_records"
        ]
        == 0
    )

    assert math.isclose(
        primary[
            "threshold_value"
        ],
        0.134,
        rel_tol=0.0,
        abs_tol=1e-15,
    )

    #
    # --------------------------------------------------
    # 4. Decision boundary uses >= exactly.
    # --------------------------------------------------
    #
    assert (
        decision_from_score(
            score=0.133999,
            threshold=0.134,
        )
        == 0
    )

    assert (
        decision_from_score(
            score=0.134,
            threshold=0.134,
        )
        == 1
    )

    assert (
        decision_from_score(
            score=0.9,
            threshold=0.134,
        )
        == 1
    )

    #
    # --------------------------------------------------
    # 5. Same frozen threshold transfers across
    #    dataset and condition.
    # --------------------------------------------------
    #
    ffpp_test = analysis_row(
        video_number=500,
        score=0.20,
        condition="H40",
        role="test",
    )

    ffpp_test_decision = (
        apply_frozen_threshold(
            analysis_record=(
                ffpp_test
            ),
            threshold_record=(
                primary
            ),
        )
    )

    assert (
        ffpp_test_decision[
            "decision"
        ]
        == 1
    )

    assert (
        ffpp_test_decision[
            "threshold_value"
        ]
        == primary[
            "threshold_value"
        ]
    )

    celeb_plt = analysis_row(
        video_number=501,
        score=0.10,
        condition="PLT",
        dataset="Celeb-DF-v2",
        role="external_evaluation",
    )

    celeb_plt_decision = (
        apply_frozen_threshold(
            analysis_record=(
                celeb_plt
            ),
            threshold_record=(
                primary
            ),
        )
    )

    assert (
        celeb_plt_decision[
            "decision"
        ]
        == 0
    )

    assert (
        celeb_plt_decision[
            "threshold_value"
        ]
        == primary[
            "threshold_value"
        ]
    )

    assert (
        celeb_plt_decision[
            "threshold_id"
        ]
        == primary[
            "threshold_id"
        ]
    )

    #
    # --------------------------------------------------
    # 6. Invalid analysis record receives no decision.
    # --------------------------------------------------
    #
    invalid = analysis_row(
        video_number=502,
        score=0.5,
        condition="RSZ",
        role="test",
        eligible=False,
    )

    invalid_decision = (
        apply_frozen_threshold(
            analysis_record=(
                invalid
            ),
            threshold_record=(
                primary
            ),
        )
    )

    assert (
        invalid_decision[
            "decision"
        ]
        is None
    )

    assert (
        invalid_decision[
            "decision_status"
        ]
        == "not_scored"
    )

    #
    # --------------------------------------------------
    # 7. Threshold is checkpoint-specific.
    # --------------------------------------------------
    #
    mismatched = copy.deepcopy(
        ffpp_test
    )

    mismatched[
        "checkpoint_sha256"
    ] = (
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    )

    checkpoint_rejection_passed = False

    try:
        apply_frozen_threshold(
            analysis_record=(
                mismatched
            ),
            threshold_record=(
                primary
            ),
        )

    except ValueError:
        checkpoint_rejection_passed = True

    assert (
        checkpoint_rejection_passed
    )

    #
    # --------------------------------------------------
    # 8. Duplicate clean validation video is forbidden.
    # --------------------------------------------------
    #
    duplicated_rows = list(
        rows
    )

    duplicated_rows.append(
        copy.deepcopy(
            rows[
                0
            ]
        )
    )

    duplicate_rejection_passed = False

    try:
        select_primary_thresholds_from_analysis_records(
            duplicated_rows,
            target_fpr=0.05,
        )

    except ValueError:
        duplicate_rejection_passed = True

    assert (
        duplicate_rejection_passed
    )

    print(
        "OPERATING-POINT CONTRACT VALIDATION"
    )
    print(
        "  target FPR:                    0.05"
    )
    print(
        "  n=140 maximum false positives: 7"
    )
    print(
        "  exact 7/140 threshold:         PASSED"
    )
    print(
        "  unattainable exact-FPR tie:    PASSED"
    )
    print(
        "  clean FF++ validation only:    PASSED"
    )
    print(
        "  real-only calibration:         PASSED"
    )
    print(
        "  processed validation ignored:  PASSED"
    )
    print(
        "  FF++ test ignored:             PASSED"
    )
    print(
        "  Celeb-DF-v2 ignored:           PASSED"
    )
    print(
        "  score == threshold -> fake:    PASSED"
    )
    print(
        "  frozen cross-condition use:    PASSED"
    )
    print(
        "  frozen cross-dataset use:      PASSED"
    )
    print(
        "  invalid record -> no decision: PASSED"
    )
    print(
        "  checkpoint mismatch rejected:  PASSED"
    )
    print(
        "  duplicate calibration rejected:PASSED"
    )
    print()
    print(
        "OPERATING-POINT CONTRACT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()