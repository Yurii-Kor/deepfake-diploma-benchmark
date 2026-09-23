from __future__ import annotations

import math

from study.analysis.metrics import (
    binary_auc,
    confusion_counts,
    false_negative_rate,
    false_positive_rate,
    fixed_threshold_metrics,
    half_total_error_rate,
)


def assert_close(
    actual,
    expected,
):
    assert math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=1e-15,
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


def main():
    #
    # --------------------------------------------------
    # 1. Perfect discrimination.
    # --------------------------------------------------
    #
    perfect_auc = binary_auc(
        real_scores=[
            0.10,
            0.20,
        ],
        manipulated_scores=[
            0.80,
            0.90,
        ],
    )

    assert_close(
        perfect_auc,
        1.0,
    )

    #
    # --------------------------------------------------
    # 2. Completely reversed discrimination.
    # --------------------------------------------------
    #
    reversed_auc = binary_auc(
        real_scores=[
            0.80,
            0.90,
        ],
        manipulated_scores=[
            0.10,
            0.20,
        ],
    )

    assert_close(
        reversed_auc,
        0.0,
    )

    #
    # --------------------------------------------------
    # 3. Complete score tie.
    #
    # Every real/manipulated comparison contributes
    # exactly 0.5.
    # --------------------------------------------------
    #
    tied_auc = binary_auc(
        real_scores=[
            0.50,
            0.50,
        ],
        manipulated_scores=[
            0.50,
            0.50,
        ],
    )

    assert_close(
        tied_auc,
        0.5,
    )

    #
    # --------------------------------------------------
    # 4. Partial ties.
    #
    # Real:
    #   0.1, 0.5
    #
    # Manipulated:
    #   0.5, 0.9
    #
    # Pairwise contributions:
    #
    #   .5 vs .1 -> 1
    #   .5 vs .5 -> .5
    #   .9 vs .1 -> 1
    #   .9 vs .5 -> 1
    #
    # total = 3.5 / 4 = .875
    # --------------------------------------------------
    #
    partial_tie_auc = binary_auc(
        real_scores=[
            0.10,
            0.50,
        ],
        manipulated_scores=[
            0.50,
            0.90,
        ],
    )

    assert_close(
        partial_tie_auc,
        0.875,
    )

    #
    # --------------------------------------------------
    # 5. Perfect fixed-threshold decisions.
    # --------------------------------------------------
    #
    perfect_metrics = (
        fixed_threshold_metrics(
            labels=[
                0,
                0,
                1,
                1,
            ],
            decisions=[
                0,
                0,
                1,
                1,
            ],
        )
    )

    assert (
        perfect_metrics[
            "tn"
        ]
        == 2
    )

    assert (
        perfect_metrics[
            "fp"
        ]
        == 0
    )

    assert (
        perfect_metrics[
            "tp"
        ]
        == 2
    )

    assert (
        perfect_metrics[
            "fn"
        ]
        == 0
    )

    assert_close(
        perfect_metrics[
            "fpr"
        ],
        0.0,
    )

    assert_close(
        perfect_metrics[
            "fnr"
        ],
        0.0,
    )

    assert_close(
        perfect_metrics[
            "hter"
        ],
        0.0,
    )

    #
    # --------------------------------------------------
    # 6. Controlled imperfect case.
    #
    # Real labels:
    #   [0, 0, 0]
    #
    # Decisions:
    #   [0, 1, 0]
    #
    # -> TN=2 FP=1
    # -> FPR=1/3
    #
    # Manipulated labels:
    #   [1, 1, 1]
    #
    # Decisions:
    #   [1, 0, 0]
    #
    # -> TP=1 FN=2
    # -> FNR=2/3
    #
    # HTER = (.333... + .666...) / 2 = .5
    # --------------------------------------------------
    #
    imperfect = (
        fixed_threshold_metrics(
            labels=[
                0,
                0,
                0,
                1,
                1,
                1,
            ],
            decisions=[
                0,
                1,
                0,
                1,
                0,
                0,
            ],
        )
    )

    assert (
        imperfect[
            "tn"
        ]
        == 2
    )

    assert (
        imperfect[
            "fp"
        ]
        == 1
    )

    assert (
        imperfect[
            "tp"
        ]
        == 1
    )

    assert (
        imperfect[
            "fn"
        ]
        == 2
    )

    assert_close(
        imperfect[
            "fpr"
        ],
        1.0 / 3.0,
    )

    assert_close(
        imperfect[
            "fnr"
        ],
        2.0 / 3.0,
    )

    assert_close(
        imperfect[
            "hter"
        ],
        0.5,
    )

    #
    # --------------------------------------------------
    # 7. Direct confusion-count primitive.
    # --------------------------------------------------
    #
    counts = confusion_counts(
        labels=[
            0,
            1,
            0,
            1,
        ],
        decisions=[
            1,
            1,
            0,
            0,
        ],
    )

    assert counts == {
        "tp": 1,
        "fp": 1,
        "tn": 1,
        "fn": 1,
    }

    #
    # --------------------------------------------------
    # 8. Direct FPR/FNR/HTER primitives.
    # --------------------------------------------------
    #
    assert_close(
        false_positive_rate(
            false_positives=1,
            true_negatives=3,
        ),
        0.25,
    )

    assert_close(
        false_negative_rate(
            false_negatives=1,
            true_positives=4,
        ),
        0.20,
    )

    assert_close(
        half_total_error_rate(
            fpr=0.25,
            fnr=0.20,
        ),
        0.225,
    )

    #
    # --------------------------------------------------
    # 9. AUC requires both classes.
    # --------------------------------------------------
    #
    assert_raises(
        ValueError,
        lambda: binary_auc(
            real_scores=[],
            manipulated_scores=[
                0.8,
            ],
        ),
    )

    assert_raises(
        ValueError,
        lambda: binary_auc(
            real_scores=[
                0.2,
            ],
            manipulated_scores=[],
        ),
    )

    #
    # --------------------------------------------------
    # 10. Non-finite / out-of-range detector scores
    #     are forbidden.
    # --------------------------------------------------
    #
    assert_raises(
        ValueError,
        lambda: binary_auc(
            real_scores=[
                float(
                    "nan"
                ),
            ],
            manipulated_scores=[
                0.8,
            ],
        ),
    )

    assert_raises(
        ValueError,
        lambda: binary_auc(
            real_scores=[
                0.2,
            ],
            manipulated_scores=[
                1.1,
            ],
        ),
    )

    #
    # --------------------------------------------------
    # 11. labels and decisions must align one-to-one.
    # --------------------------------------------------
    #
    assert_raises(
        ValueError,
        lambda: confusion_counts(
            labels=[
                0,
                1,
            ],
            decisions=[
                0,
            ],
        ),
    )

    #
    # --------------------------------------------------
    # 12. Non-binary labels and decisions are rejected.
    # --------------------------------------------------
    #
    assert_raises(
        ValueError,
        lambda: confusion_counts(
            labels=[
                0,
                2,
            ],
            decisions=[
                0,
                1,
            ],
        ),
    )

    assert_raises(
        ValueError,
        lambda: confusion_counts(
            labels=[
                0,
                1,
            ],
            decisions=[
                0,
                -1,
            ],
        ),
    )

    #
    # --------------------------------------------------
    # 13. FPR is undefined without real observations.
    # --------------------------------------------------
    #
    assert_raises(
        ValueError,
        lambda: false_positive_rate(
            false_positives=0,
            true_negatives=0,
        ),
    )

    #
    # --------------------------------------------------
    # 14. FNR is undefined without manipulated
    #     observations.
    # --------------------------------------------------
    #
    assert_raises(
        ValueError,
        lambda: false_negative_rate(
            false_negatives=0,
            true_positives=0,
        ),
    )

    #
    # --------------------------------------------------
    # 15. HTER inputs must be valid rates.
    # --------------------------------------------------
    #
    assert_raises(
        ValueError,
        lambda: half_total_error_rate(
            fpr=-0.1,
            fnr=0.2,
        ),
    )

    assert_raises(
        ValueError,
        lambda: half_total_error_rate(
            fpr=0.1,
            fnr=1.1,
        ),
    )

    print(
        "METRIC PRIMITIVES VALIDATION"
    )
    print(
        "  perfect AUC = 1.0:             PASSED"
    )
    print(
        "  reversed AUC = 0.0:            PASSED"
    )
    print(
        "  complete ties AUC = 0.5:       PASSED"
    )
    print(
        "  partial-tie AUC = 0.875:       PASSED"
    )
    print(
        "  perfect FPR/FNR/HTER:          PASSED"
    )
    print(
        "  imperfect confusion counts:    PASSED"
    )
    print(
        "  imperfect FPR = 1/3:           PASSED"
    )
    print(
        "  imperfect FNR = 2/3:           PASSED"
    )
    print(
        "  imperfect HTER = 0.5:          PASSED"
    )
    print(
        "  manipulated class positive:    PASSED"
    )
    print(
        "  score validation:              PASSED"
    )
    print(
        "  binary label validation:       PASSED"
    )
    print(
        "  undefined-rate rejection:      PASSED"
    )
    print()
    print(
        "METRIC PRIMITIVES VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()