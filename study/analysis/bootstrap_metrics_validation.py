from __future__ import annotations

import copy
import math

from study.analysis.bootstrap_metrics import (
    bootstrap_absolute_metrics,
    bootstrap_diagnostic_threshold_displacement,
    bootstrap_paired_degradation_metrics,
    derive_stream_seed,
    percentile_interval,
)


CHECKPOINT = (
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)


def assert_close(
    actual,
    expected,
    *,
    tolerance=1e-12,
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


def validation_row(
    *,
    unit_id,
    condition,
    score,
    eligible=True,
):
    return {
        "detector": "xception",

        "checkpoint_sha256": (
            CHECKPOINT
        ),

        "dataset": (
            "FaceForensics++"
        ),

        "role": (
            "validation"
        ),

        "study_label": 0,

        "manipulation": "",

        "source_video_id": (
            unit_id
        ),

        "base_video_id": (
            unit_id
        ),

        "pairing_uid": (
            "FaceForensics++::original/{}.mp4"
            .format(
                unit_id
            )
        ),

        "relative_source_path": (
            "original/{}.mp4"
            .format(
                unit_id
            )
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


def evaluation_row(
    *,
    unit_id,
    condition,
    label,
    score,
    manipulation="",
):
    if label == 0:
        path = (
            "original/{}.mp4"
            .format(
                unit_id
            )
        )
    else:
        path = (
            "{}/{}_target.mp4"
            .format(
                manipulation,
                unit_id,
            )
        )

    return {
        "detector": "xception",

        "checkpoint_sha256": (
            CHECKPOINT
        ),

        "dataset": (
            "FaceForensics++"
        ),

        "role": (
            "test"
        ),

        "study_label": (
            label
        ),

        "manipulation": (
            manipulation
        ),

        "source_video_id": (
            unit_id
        ),

        "base_video_id": (
            unit_id
        ),

        "pairing_uid": (
            "FaceForensics++::{}"
            .format(
                path
            )
        ),

        "relative_source_path": (
            path
        ),

        "condition": (
            condition
        ),

        "analysis_eligible": True,
        "analysis_status": (
            "valid_score"
        ),

        "video_score": (
            score
        ),
    }


def main():
    #
    # --------------------------------------------------
    # 1. Independent deterministic RNG streams.
    # --------------------------------------------------
    #
    calibration_seed = (
        derive_stream_seed(
            seed=1024,
            stream_name=(
                "primary_clean_validation"
            ),
        )
    )

    evaluation_seed = (
        derive_stream_seed(
            seed=1024,
            stream_name=(
                "absolute_evaluation"
            ),
        )
    )

    assert (
        calibration_seed
        != evaluation_seed
    )

    assert (
        calibration_seed
        == derive_stream_seed(
            seed=1024,
            stream_name=(
                "primary_clean_validation"
            ),
        )
    )

    #
    # --------------------------------------------------
    # 2. Percentile interval implementation.
    #
    # values = 0,1,2,3,4
    # 95% tails = .025 and .975
    #
    # positions:
    #   .025 * 4 = .1  -> .1
    #   .975 * 4 = 3.9 -> 3.9
    # --------------------------------------------------
    #
    interval = percentile_interval(
        [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
        ],
        ci_level=0.95,
    )

    assert_close(
        interval[
            "lower"
        ],
        0.1,
    )

    assert_close(
        interval[
            "upper"
        ],
        3.9,
    )

    assert (
        interval[
            "replicate_count"
        ]
        == 5
    )

    #
    # --------------------------------------------------
    # 3. Clean validation calibration population.
    #
    # Twenty unique real videos.
    # --------------------------------------------------
    #
    clean_validation = [
        validation_row(
            unit_id=(
                "{:03d}".format(
                    index
                )
            ),
            condition="CLN",
            score=(
                index
                / 100.0
            ),
        )
        for index in range(
            1,
            21,
        )
    ]

    #
    # --------------------------------------------------
    # 4. Perfect absolute evaluation.
    #
    # Every real score = 0.
    # Every fake score = 1.
    #
    # Regardless of bootstrap multiplicity and
    # replicate-specific clean threshold:
    #
    #   AUC  = 1
    #   FPR  = 0
    #   FNR  = 0
    #   HTER = 0
    # --------------------------------------------------
    #
    absolute_real = [
        evaluation_row(
            unit_id=(
                "R{}".format(
                    index
                )
            ),
            condition="CLN",
            label=0,
            score=0.0,
        )
        for index in range(
            1,
            5,
        )
    ]

    absolute_fake = [
        evaluation_row(
            unit_id=(
                "F{}".format(
                    index
                )
            ),
            condition="CLN",
            label=1,
            manipulation="DeepFakes",
            score=1.0,
        )
        for index in range(
            1,
            5,
        )
    ]

    absolute = (
        bootstrap_absolute_metrics(
            real_rows=(
                absolute_real
            ),
            manipulated_rows=(
                absolute_fake
            ),
            clean_validation_real_rows=(
                clean_validation
            ),
            replicates=100,
            seed=1024,
            stream_name=(
                "validation_absolute_test"
            ),
        )
    )

    assert (
        absolute[
            "replicate_count"
        ]
        == 100
    )

    for replicate in (
        absolute[
            "replicates"
        ]
    ):
        assert_close(
            replicate[
                "auc"
            ],
            1.0,
        )

        assert_close(
            replicate[
                "fpr"
            ],
            0.0,
        )

        assert_close(
            replicate[
                "fnr"
            ],
            0.0,
        )

        assert_close(
            replicate[
                "hter"
            ],
            0.0,
        )

    #
    # Calibration threshold is genuinely
    # re-estimated rather than frozen.
    #
    threshold_values = {
        replicate[
            "threshold"
        ]
        for replicate in (
            absolute[
                "replicates"
            ]
        )
    }

    assert (
        len(
            threshold_values
        )
        > 1
    )

    #
    # Same seed and same stream => identical
    # bootstrap result.
    #
    absolute_again = (
        bootstrap_absolute_metrics(
            real_rows=(
                absolute_real
            ),
            manipulated_rows=(
                absolute_fake
            ),
            clean_validation_real_rows=(
                clean_validation
            ),
            replicates=100,
            seed=1024,
            stream_name=(
                "validation_absolute_test"
            ),
        )
    )

    assert (
        absolute
        == absolute_again
    )

    #
    # --------------------------------------------------
    # 5. Controlled paired degradation.
    #
    # CLN:
    #   real = 0
    #   fake = 1
    #
    # RSZ:
    #   real = 1
    #   fake = 0
    #
    # Therefore every replicate has:
    #
    #   clean AUC  = 1
    #   RSZ AUC    = 0
    #   delta AUC  = -1
    #
    # Since calibration threshold comes from scores
    # .01-.20:
    #
    #   clean FPR/FNR/HTER = 0
    #   RSZ FPR/FNR/HTER   = 1
    # --------------------------------------------------
    #
    clean_real = []
    degraded_real = []

    clean_fake = []
    degraded_fake = []

    for index in range(
        1,
        5,
    ):
        unit_id = (
            "R{}".format(
                index
            )
        )

        clean_real.append(
            evaluation_row(
                unit_id=unit_id,
                condition="CLN",
                label=0,
                score=0.0,
            )
        )

        degraded_real.append(
            evaluation_row(
                unit_id=unit_id,
                condition="RSZ",
                label=0,
                score=1.0,
            )
        )

    for index in range(
        1,
        5,
    ):
        unit_id = (
            "F{}".format(
                index
            )
        )

        clean_fake.append(
            evaluation_row(
                unit_id=unit_id,
                condition="CLN",
                label=1,
                manipulation="DeepFakes",
                score=1.0,
            )
        )

        degraded_fake.append(
            evaluation_row(
                unit_id=unit_id,
                condition="RSZ",
                label=1,
                manipulation="DeepFakes",
                score=0.0,
            )
        )

    paired = (
        bootstrap_paired_degradation_metrics(
            clean_real_rows=(
                clean_real
            ),
            clean_manipulated_rows=(
                clean_fake
            ),
            degraded_real_rows=(
                degraded_real
            ),
            degraded_manipulated_rows=(
                degraded_fake
            ),
            clean_validation_real_rows=(
                clean_validation
            ),
            replicates=100,
            seed=1024,
            stream_name=(
                "validation_paired_test"
            ),
        )
    )

    assert (
        paired[
            "n_paired_real_units"
        ]
        == 4
    )

    assert (
        paired[
            "n_paired_manipulated_units"
        ]
        == 4
    )

    for replicate in (
        paired[
            "replicates"
        ]
    ):
        assert_close(
            replicate[
                "clean_auc"
            ],
            1.0,
        )

        assert_close(
            replicate[
                "degraded_auc"
            ],
            0.0,
        )

        assert_close(
            replicate[
                "delta_auc"
            ],
            -1.0,
        )

        assert_close(
            replicate[
                "clean_fpr"
            ],
            0.0,
        )

        assert_close(
            replicate[
                "degraded_fpr"
            ],
            1.0,
        )

        assert_close(
            replicate[
                "delta_fpr"
            ],
            1.0,
        )

        assert_close(
            replicate[
                "clean_fnr"
            ],
            0.0,
        )

        assert_close(
            replicate[
                "degraded_fnr"
            ],
            1.0,
        )

        assert_close(
            replicate[
                "delta_fnr"
            ],
            1.0,
        )

        assert_close(
            replicate[
                "clean_hter"
            ],
            0.0,
        )

        assert_close(
            replicate[
                "degraded_hter"
            ],
            1.0,
        )

        assert_close(
            replicate[
                "delta_hter"
            ],
            1.0,
        )

    #
    # --------------------------------------------------
    # 6. Missing paired evaluation unit rejected.
    #
    # The delta bootstrap must not silently compare
    # different CLN and degraded populations.
    # --------------------------------------------------
    #
    missing_pair = copy.deepcopy(
        degraded_fake
    )

    missing_pair.pop()

    assert_raises(
        ValueError,
        lambda: (
            bootstrap_paired_degradation_metrics(
                clean_real_rows=(
                    clean_real
                ),
                clean_manipulated_rows=(
                    clean_fake
                ),
                degraded_real_rows=(
                    degraded_real
                ),
                degraded_manipulated_rows=(
                    missing_pair
                ),
                clean_validation_real_rows=(
                    clean_validation
                ),
                replicates=10,
                seed=1024,
            )
        ),
    )

    #
    # --------------------------------------------------
    # 7. Diagnostic threshold displacement bootstrap.
    #
    # Degraded validation scores are exactly clean + .1.
    #
    # Both thresholds are re-estimated from the SAME
    # sampled IDs, so displacement must remain .1,
    # up to representable floating-point boundary drift.
    # --------------------------------------------------
    #
    degraded_validation = [
        validation_row(
            unit_id=(
                "{:03d}".format(
                    index
                )
            ),
            condition="RSZ",
            score=(
                index
                / 100.0
                + 0.10
            ),
        )
        for index in range(
            1,
            21,
        )
    ]

    diagnostic = (
        bootstrap_diagnostic_threshold_displacement(
            clean_validation_rows=(
                clean_validation
            ),
            degraded_validation_rows=(
                degraded_validation
            ),
            degraded_condition="RSZ",
            replicates=100,
            seed=1024,
        )
    )

    assert (
        diagnostic[
            "n_clean_valid_real"
        ]
        == 20
    )

    assert (
        diagnostic[
            "n_degraded_valid_real"
        ]
        == 20
    )

    assert (
        diagnostic[
            "n_paired_valid_real"
        ]
        == 20
    )

    for replicate in (
        diagnostic[
            "replicates"
        ]
    ):
        assert_close(
            replicate[
                "threshold_displacement"
            ],
            0.10,
            tolerance=1e-12,
        )

    #
    # --------------------------------------------------
    # 8. Diagnostic bootstrap uses paired valid
    #    validation IDs.
    #
    # Make one RSZ real video invalid. Clean still has
    # 20 valid, RSZ has 19, paired universe must be 19.
    # --------------------------------------------------
    #
    degraded_with_failure = (
        copy.deepcopy(
            degraded_validation
        )
    )

    degraded_with_failure[
        -1
    ][
        "analysis_eligible"
    ] = False

    degraded_with_failure[
        -1
    ][
        "analysis_status"
    ] = "invalid_inference"

    degraded_with_failure[
        -1
    ][
        "video_score"
    ] = None

    diagnostic_failure = (
        bootstrap_diagnostic_threshold_displacement(
            clean_validation_rows=(
                clean_validation
            ),
            degraded_validation_rows=(
                degraded_with_failure
            ),
            degraded_condition="RSZ",
            replicates=25,
            seed=1024,
        )
    )

    assert (
        diagnostic_failure[
            "n_clean_valid_real"
        ]
        == 20
    )

    assert (
        diagnostic_failure[
            "n_degraded_valid_real"
        ]
        == 19
    )

    assert (
        diagnostic_failure[
            "n_paired_valid_real"
        ]
        == 19
    )

    #
    # --------------------------------------------------
    # 9. Constant bootstrap distribution gives
    #    degenerate percentile interval.
    # --------------------------------------------------
    #
    auc_interval = (
        percentile_interval(
            [
                replicate[
                    "auc"
                ]
                for replicate in (
                    absolute[
                        "replicates"
                    ]
                )
            ],
            ci_level=0.95,
        )
    )

    assert_close(
        auc_interval[
            "lower"
        ],
        1.0,
    )

    assert_close(
        auc_interval[
            "upper"
        ],
        1.0,
    )

    print(
        "BOOTSTRAP METRIC CONTRACT VALIDATION"
    )
    print(
        "  independent RNG streams:       PASSED"
    )
    print(
        "  percentile CI interpolation:   PASSED"
    )
    print(
        "  threshold re-estimation:       PASSED"
    )
    print(
        "  deterministic bootstrap:       PASSED"
    )
    print(
        "  absolute AUC bootstrap:        PASSED"
    )
    print(
        "  absolute FPR/FNR/HTER:         PASSED"
    )
    print(
        "  paired CLN/degraded sampling:  PASSED"
    )
    print(
        "  paired delta AUC:              PASSED"
    )
    print(
        "  paired delta FPR/FNR/HTER:     PASSED"
    )
    print(
        "  missing-pair rejection:        PASSED"
    )
    print(
        "  diagnostic paired bootstrap:   PASSED"
    )
    print(
        "  diagnostic displacement:       PASSED"
    )
    print(
        "  diagnostic failure intersection:PASSED"
    )
    print(
        "  95% percentile CI:             PASSED"
    )
    print()
    print(
        "BOOTSTRAP METRIC CONTRACT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()