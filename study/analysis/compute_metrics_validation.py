from __future__ import annotations

import copy
import math

from study.analysis.compute_metrics import (
    compute_absolute_metrics,
    compute_paired_degradation_metrics,
)


CHECKPOINT = (
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)

THRESHOLD_ID = (
    "xception_ffpp_validation_cln_fpr005_aaaaaaaaaaaa"
)

THRESHOLD = 0.5

MANIPULATIONS = (
    "DeepFakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
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


def record(
    *,
    path,
    label,
    condition,
    score,
    manipulation="",
    eligible=True,
):
    if eligible:
        decision = (
            1
            if score >= THRESHOLD
            else 0
        )

        stored_score = (
            score
        )

        decision_status = (
            "decided"
        )

    else:
        decision = None
        stored_score = None
        decision_status = (
            "not_scored"
        )

    return {
        "schema_version": 1,

        "detector": "xception",
        "checkpoint_sha256": (
            CHECKPOINT
        ),

        "dataset": (
            "FaceForensics++"
        ),

        "role": "test",

        "pairing_uid": (
            "FaceForensics++::{}".format(
                path
            )
        ),

        "relative_source_path": (
            path
        ),

        "base_video_id": (
            path
        ),

        "source_video_id": (
            path
        ),

        "study_label": (
            label
        ),

        "manipulation": (
            manipulation
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
            stored_score
        ),

        "threshold_id": (
            THRESHOLD_ID
        ),

        "threshold_value": (
            THRESHOLD
        ),

        "decision_status": (
            decision_status
        ),

        "decision": (
            decision
        ),
    }


def find_absolute(
    results,
    *,
    condition,
    group_name,
):
    matches = [
        row
        for row in results
        if (
            row[
                "condition"
            ]
            == condition
            and row[
                "group_name"
            ]
            == group_name
        )
    ]

    assert len(
        matches
    ) == 1

    return matches[
        0
    ]


def find_delta(
    results,
    *,
    degraded_condition,
    group_name,
):
    matches = [
        row
        for row in results
        if (
            row[
                "degraded_condition"
            ]
            == degraded_condition
            and row[
                "group_name"
            ]
            == group_name
        )
    ]

    assert len(
        matches
    ) == 1

    return matches[
        0
    ]


def build_complete_fixture():
    rows = []

    #
    # --------------------------------------------------
    # Real videos.
    #
    # CLN:
    #   .1, .4 -> both real at threshold .5
    #
    # RSZ:
    #   .6, .2 -> one FP, one TN
    #
    # FPR:
    #   CLN = 0
    #   RSZ = .5
    # --------------------------------------------------
    #
    rows.extend(
        [
            record(
                path="original/001.mp4",
                label=0,
                condition="CLN",
                score=0.10,
            ),
            record(
                path="original/002.mp4",
                label=0,
                condition="CLN",
                score=0.40,
            ),
            record(
                path="original/001.mp4",
                label=0,
                condition="RSZ",
                score=0.60,
            ),
            record(
                path="original/002.mp4",
                label=0,
                condition="RSZ",
                score=0.20,
            ),
        ]
    )

    for manipulation in (
        MANIPULATIONS
    ):
        prefix = (
            manipulation.lower()
        )

        #
        # All CLN fake videos:
        #   .8, .9
        #
        # => perfect AUC
        # => FNR = 0
        #
        rows.extend(
            [
                record(
                    path=(
                        "{}/001_101.mp4"
                        .format(
                            prefix
                        )
                    ),
                    label=1,
                    manipulation=(
                        manipulation
                    ),
                    condition="CLN",
                    score=0.80,
                ),
                record(
                    path=(
                        "{}/002_102.mp4"
                        .format(
                            prefix
                        )
                    ),
                    label=1,
                    manipulation=(
                        manipulation
                    ),
                    condition="CLN",
                    score=0.90,
                ),
            ]
        )

        if (
            manipulation
            == "DeepFakes"
        ):
            #
            # RSZ DeepFakes:
            #   .7 -> TP
            #   .3 -> FN
            #
            # Against real [.6, .2]:
            #
            # .7 beats both real scores = 2
            # .3 beats .2 only          = 1
            #
            # AUC = 3 / 4 = .75
            #
            # FNR = .5
            # FPR = .5
            # HTER = .5
            #
            fake_scores = (
                0.70,
                0.30,
            )

        else:
            #
            # Other methods remain perfectly ranked.
            #
            fake_scores = (
                0.80,
                0.90,
            )

        rows.extend(
            [
                record(
                    path=(
                        "{}/001_101.mp4"
                        .format(
                            prefix
                        )
                    ),
                    label=1,
                    manipulation=(
                        manipulation
                    ),
                    condition="RSZ",
                    score=(
                        fake_scores[
                            0
                        ]
                    ),
                ),
                record(
                    path=(
                        "{}/002_102.mp4"
                        .format(
                            prefix
                        )
                    ),
                    label=1,
                    manipulation=(
                        manipulation
                    ),
                    condition="RSZ",
                    score=(
                        fake_scores[
                            1
                        ]
                    ),
                ),
            ]
        )

    return rows


def main():
    rows = (
        build_complete_fixture()
    )

    #
    # --------------------------------------------------
    # 1. Absolute metric computation.
    #
    # Two conditions × five FF++ groups:
    #
    #   DeepFakes
    #   Face2Face
    #   FaceSwap
    #   NeuralTextures
    #   pooled
    #
    # --------------------------------------------------
    #
    absolute = (
        compute_absolute_metrics(
            rows
        )
    )

    assert len(
        absolute
    ) == 10

    cln_df = find_absolute(
        absolute,
        condition="CLN",
        group_name="DeepFakes",
    )

    assert_close(
        cln_df[
            "auc"
        ],
        1.0,
    )

    assert_close(
        cln_df[
            "fpr"
        ],
        0.0,
    )

    assert_close(
        cln_df[
            "fnr"
        ],
        0.0,
    )

    assert_close(
        cln_df[
            "hter"
        ],
        0.0,
    )

    assert (
        cln_df[
            "tn"
        ]
        == 2
    )

    assert (
        cln_df[
            "tp"
        ]
        == 2
    )

    #
    # --------------------------------------------------
    # 2. Controlled imperfect RSZ DeepFakes result.
    # --------------------------------------------------
    #
    rsz_df = find_absolute(
        absolute,
        condition="RSZ",
        group_name="DeepFakes",
    )

    assert_close(
        rsz_df[
            "auc"
        ],
        0.75,
    )

    assert_close(
        rsz_df[
            "fpr"
        ],
        0.50,
    )

    assert_close(
        rsz_df[
            "fnr"
        ],
        0.50,
    )

    assert_close(
        rsz_df[
            "hter"
        ],
        0.50,
    )

    assert (
        rsz_df[
            "fp"
        ]
        == 1
    )

    assert (
        rsz_df[
            "tn"
        ]
        == 1
    )

    assert (
        rsz_df[
            "tp"
        ]
        == 1
    )

    assert (
        rsz_df[
            "fn"
        ]
        == 1
    )

    #
    # --------------------------------------------------
    # 3. Pooled RSZ.
    #
    # There are 8 fake videos total.
    #
    # Seven are TP; one DeepFakes video is FN:
    #
    # FNR = 1/8 = .125
    #
    # Real FPR = .5
    #
    # HTER = (.5 + .125)/2 = .3125
    #
    # AUC:
    #
    # DeepFakes contributes 3 pairwise wins.
    # Other 6 fake videos contribute 12 wins.
    #
    # 15 / 16 = .9375
    # --------------------------------------------------
    #
    rsz_pooled = find_absolute(
        absolute,
        condition="RSZ",
        group_name="pooled",
    )

    assert (
        rsz_pooled[
            "n_real_valid"
        ]
        == 2
    )

    assert (
        rsz_pooled[
            "n_manipulated_valid"
        ]
        == 8
    )

    assert_close(
        rsz_pooled[
            "auc"
        ],
        0.9375,
    )

    assert_close(
        rsz_pooled[
            "fpr"
        ],
        0.50,
    )

    assert_close(
        rsz_pooled[
            "fnr"
        ],
        0.125,
    )

    assert_close(
        rsz_pooled[
            "hter"
        ],
        0.3125,
    )

    #
    # --------------------------------------------------
    # 4. Paired clean -> degraded metrics.
    # --------------------------------------------------
    #
    paired = (
        compute_paired_degradation_metrics(
            rows
        )
    )

    assert len(
        paired
    ) == 5

    df_delta = find_delta(
        paired,
        degraded_condition="RSZ",
        group_name="DeepFakes",
    )

    assert (
        df_delta[
            "n_real_paired"
        ]
        == 2
    )

    assert (
        df_delta[
            "n_manipulated_paired"
        ]
        == 2
    )

    assert_close(
        df_delta[
            "clean_auc"
        ],
        1.0,
    )

    assert_close(
        df_delta[
            "degraded_auc"
        ],
        0.75,
    )

    assert_close(
        df_delta[
            "delta_auc"
        ],
        -0.25,
    )

    assert_close(
        df_delta[
            "delta_fpr"
        ],
        0.50,
    )

    assert_close(
        df_delta[
            "delta_fnr"
        ],
        0.50,
    )

    assert_close(
        df_delta[
            "delta_hter"
        ],
        0.50,
    )

    pooled_delta = find_delta(
        paired,
        degraded_condition="RSZ",
        group_name="pooled",
    )

    assert_close(
        pooled_delta[
            "clean_auc"
        ],
        1.0,
    )

    assert_close(
        pooled_delta[
            "degraded_auc"
        ],
        0.9375,
    )

    assert_close(
        pooled_delta[
            "delta_auc"
        ],
        -0.0625,
    )

    assert_close(
        pooled_delta[
            "delta_fpr"
        ],
        0.50,
    )

    assert_close(
        pooled_delta[
            "delta_fnr"
        ],
        0.125,
    )

    assert_close(
        pooled_delta[
            "delta_hter"
        ],
        0.3125,
    )

    #
    # --------------------------------------------------
    # 5. Explicit regression: paired delta must
    #    recompute BOTH sides on the valid intersection.
    #
    # Make original/002 invalid only under RSZ.
    #
    # Also make its CLN score .6, so the unpaired CLN
    # FPR would be .5.
    #
    # Since that video is absent from the RSZ side,
    # paired CLN must use only original/001 and therefore
    # paired clean FPR must be 0.
    # --------------------------------------------------
    #
    intersection_rows = (
        copy.deepcopy(
            rows
        )
    )

    clean_removed_counterpart = next(
        row
        for row in intersection_rows
        if (
            row[
                "condition"
            ]
            == "CLN"
            and row[
                "relative_source_path"
            ]
            == "original/002.mp4"
        )
    )

    clean_removed_counterpart[
        "video_score"
    ] = 0.60

    clean_removed_counterpart[
        "decision"
    ] = 1

    degraded_invalid = next(
        row
        for row in intersection_rows
        if (
            row[
                "condition"
            ]
            == "RSZ"
            and row[
                "relative_source_path"
            ]
            == "original/002.mp4"
        )
    )

    degraded_invalid[
        "analysis_eligible"
    ] = False

    degraded_invalid[
        "analysis_status"
    ] = "invalid_inference"

    degraded_invalid[
        "video_score"
    ] = None

    degraded_invalid[
        "decision_status"
    ] = "not_scored"

    degraded_invalid[
        "decision"
    ] = None

    paired_intersection = (
        compute_paired_degradation_metrics(
            intersection_rows
        )
    )

    intersection_df = find_delta(
        paired_intersection,
        degraded_condition="RSZ",
        group_name="DeepFakes",
    )

    assert (
        intersection_df[
            "n_real_clean_valid"
        ]
        == 2
    )

    assert (
        intersection_df[
            "n_real_degraded_valid"
        ]
        == 1
    )

    assert (
        intersection_df[
            "n_real_paired"
        ]
        == 1
    )

    #
    # Critical check:
    #
    # CLN metric was recomputed using only the single
    # real video that also survived RSZ.
    #
    assert_close(
        intersection_df[
            "clean_fpr"
        ],
        0.0,
    )

    assert_close(
        intersection_df[
            "degraded_fpr"
        ],
        1.0,
    )

    #
    # --------------------------------------------------
    # 6. Stored decision must agree with frozen
    #    threshold. Tampering is rejected.
    # --------------------------------------------------
    #
    bad_decision_rows = (
        copy.deepcopy(
            rows
        )
    )

    bad_decision = next(
        row
        for row in bad_decision_rows
        if (
            row[
                "condition"
            ]
            == "CLN"
            and row[
                "relative_source_path"
            ]
            == "original/001.mp4"
        )
    )

    #
    # score=.1 < threshold=.5, therefore decision must
    # be 0. Deliberately corrupt it.
    #
    bad_decision[
        "decision"
    ] = 1

    assert_raises(
        ValueError,
        lambda: (
            compute_absolute_metrics(
                bad_decision_rows
            )
        ),
    )

    #
    # --------------------------------------------------
    # 7. Mixed frozen thresholds inside one metric
    #    population are rejected.
    # --------------------------------------------------
    #
    bad_threshold_rows = (
        copy.deepcopy(
            rows
        )
    )

    bad_threshold = next(
        row
        for row in bad_threshold_rows
        if (
            row[
                "condition"
            ]
            == "CLN"
            and row[
                "study_label"
            ]
            == 1
        )
    )

    bad_threshold[
        "threshold_value"
    ] = 0.55

    assert_raises(
        ValueError,
        lambda: (
            compute_absolute_metrics(
                bad_threshold_rows
            )
        ),
    )

    print(
        "METRIC COMPUTATION CONTRACT VALIDATION"
    )
    print(
        "  absolute FF++ metric groups:   PASSED"
    )
    print(
        "  perfect CLN metrics:           PASSED"
    )
    print(
        "  controlled RSZ AUC = .75:      PASSED"
    )
    print(
        "  controlled RSZ FPR = .5:       PASSED"
    )
    print(
        "  controlled RSZ FNR = .5:       PASSED"
    )
    print(
        "  controlled RSZ HTER = .5:      PASSED"
    )
    print(
        "  pooled RSZ AUC = .9375:        PASSED"
    )
    print(
        "  pooled RSZ FNR = .125:         PASSED"
    )
    print(
        "  paired delta AUC:              PASSED"
    )
    print(
        "  paired delta FPR/FNR/HTER:     PASSED"
    )
    print(
        "  pairwise valid intersection:   PASSED"
    )
    print(
        "  CLN recomputed on intersection:PASSED"
    )
    print(
        "  decision/threshold consistency:PASSED"
    )
    print(
        "  threshold identity consistency:PASSED"
    )
    print()
    print(
        "METRIC COMPUTATION CONTRACT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()