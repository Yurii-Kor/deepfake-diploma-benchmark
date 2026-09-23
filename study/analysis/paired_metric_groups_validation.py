from __future__ import annotations

import copy

from study.analysis.paired_metric_groups import (
    build_paired_metric_groups,
)


CHECKPOINT = (
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)

THRESHOLD_ID = (
    "xception_ffpp_validation_cln_fpr005_aaaaaaaaaaaa"
)

THRESHOLD = 0.50


def record(
    *,
    path,
    label,
    condition,
    manipulation="",
    eligible=True,
    score=None,
    decision=None,
):
    if score is None:
        if eligible:
            score = (
                0.10
                if label == 0
                else 0.90
            )
        else:
            score = None

    if decision is None:
        if eligible:
            decision = (
                1
                if score >= THRESHOLD
                else 0
            )
        else:
            decision = None

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
            score
        ),

        "decision_status": (
            "decided"
            if eligible
            else "not_scored"
        ),

        "decision": (
            decision
        ),

        "threshold_id": (
            THRESHOLD_ID
        ),

        "threshold_value": (
            THRESHOLD
        ),
    }


def get_group(
    groups,
    *,
    group_name,
    degraded_condition,
):
    matches = [
        group
        for group in groups
        if (
            group[
                "group_name"
            ]
            == group_name
            and group[
                "degraded_condition"
            ]
            == degraded_condition
        )
    ]

    assert len(
        matches
    ) == 1

    return matches[
        0
    ]


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
    # Synthetic FF++ population:
    #
    # real:
    #   original/001.mp4
    #   original/002.mp4
    #   original/003.mp4
    #
    # DeepFakes:
    #   001_101
    #   002_102
    #   003_103
    #
    # CLN: all valid
    #
    # RSZ:
    #   original/002 invalid
    #   Deepfakes/003_103 invalid
    #
    # H40:
    #   all valid
    #
    # Therefore RSZ pairwise intersection must contain:
    #   2 real
    #   2 DeepFakes
    #
    # H40:
    #   3 real
    #   3 DeepFakes
    # --------------------------------------------------
    #
    rows = []

    real_paths = (
        "original/001.mp4",
        "original/002.mp4",
        "original/003.mp4",
    )

    fake_paths = (
        "Deepfakes/001_101.mp4",
        "Deepfakes/002_102.mp4",
        "Deepfakes/003_103.mp4",
    )

    for path in real_paths:
        rows.append(
            record(
                path=path,
                label=0,
                condition="CLN",
            )
        )

    for path in fake_paths:
        rows.append(
            record(
                path=path,
                label=1,
                manipulation="DeepFakes",
                condition="CLN",
            )
        )

    for path in real_paths:
        rows.append(
            record(
                path=path,
                label=0,
                condition="RSZ",
                eligible=(
                    path
                    != "original/002.mp4"
                ),
            )
        )

    for path in fake_paths:
        rows.append(
            record(
                path=path,
                label=1,
                manipulation="DeepFakes",
                condition="RSZ",
                eligible=(
                    path
                    != "Deepfakes/003_103.mp4"
                ),
            )
        )

    for path in real_paths:
        rows.append(
            record(
                path=path,
                label=0,
                condition="H40",
            )
        )

    for path in fake_paths:
        rows.append(
            record(
                path=path,
                label=1,
                manipulation="DeepFakes",
                condition="H40",
            )
        )

    groups = (
        build_paired_metric_groups(
            rows
        )
    )

    #
    # build_primary_metric_groups creates all five
    # FF++ group names, even when some manipulation
    # groups contain zero fake records.
    #
    # We therefore inspect DeepFakes and pooled,
    # which are the populated relevant groups here.
    #
    rsz_df = get_group(
        groups,
        group_name="DeepFakes",
        degraded_condition="RSZ",
    )

    assert (
        rsz_df[
            "n_real_clean_valid"
        ]
        == 3
    )

    assert (
        rsz_df[
            "n_real_degraded_valid"
        ]
        == 2
    )

    assert (
        rsz_df[
            "n_real_paired"
        ]
        == 2
    )

    assert (
        rsz_df[
            "n_manipulated_clean_valid"
        ]
        == 3
    )

    assert (
        rsz_df[
            "n_manipulated_degraded_valid"
        ]
        == 2
    )

    assert (
        rsz_df[
            "n_manipulated_paired"
        ]
        == 2
    )

    assert {
        row[
            "pairing_uid"
        ]
        for row in (
            rsz_df[
                "clean_real_records"
            ]
        )
    } == {
        row[
            "pairing_uid"
        ]
        for row in (
            rsz_df[
                "degraded_real_records"
            ]
        )
    }

    assert {
        row[
            "pairing_uid"
        ]
        for row in (
            rsz_df[
                "clean_manipulated_records"
            ]
        )
    } == {
        row[
            "pairing_uid"
        ]
        for row in (
            rsz_df[
                "degraded_manipulated_records"
            ]
        )
    }

    assert (
        rsz_df[
            "n_real_excluded_from_pair"
        ]
        == 1
    )

    assert (
        rsz_df[
            "n_manipulated_excluded_from_pair"
        ]
        == 1
    )

    h40_df = get_group(
        groups,
        group_name="DeepFakes",
        degraded_condition="H40",
    )

    assert (
        h40_df[
            "n_real_paired"
        ]
        == 3
    )

    assert (
        h40_df[
            "n_manipulated_paired"
        ]
        == 3
    )

    pooled_rsz = get_group(
        groups,
        group_name="pooled",
        degraded_condition="RSZ",
    )

    assert (
        pooled_rsz[
            "n_real_paired"
        ]
        == 2
    )

    assert (
        pooled_rsz[
            "n_manipulated_paired"
        ]
        == 2
    )

    #
    # --------------------------------------------------
    # Pair order must be identical between CLN and
    # degraded records.
    # --------------------------------------------------
    #
    assert [
        row[
            "pairing_uid"
        ]
        for row in (
            rsz_df[
                "clean_real_records"
            ]
        )
    ] == [
        row[
            "pairing_uid"
        ]
        for row in (
            rsz_df[
                "degraded_real_records"
            ]
        )
    ]

    assert [
        row[
            "pairing_uid"
        ]
        for row in (
            rsz_df[
                "clean_manipulated_records"
            ]
        )
    ] == [
        row[
            "pairing_uid"
        ]
        for row in (
            rsz_df[
                "degraded_manipulated_records"
            ]
        )
    ]

    #
    # --------------------------------------------------
    # Threshold must be frozen across conditions.
    # --------------------------------------------------
    #
    threshold_change_rows = copy.deepcopy(
        rows
    )

    changed = next(
        row
        for row in threshold_change_rows
        if (
            row[
                "condition"
            ]
            == "RSZ"
            and row[
                "analysis_eligible"
            ]
            is True
        )
    )

    changed[
        "threshold_value"
    ] = 0.60

    assert_raises(
        ValueError,
        lambda: (
            build_paired_metric_groups(
                threshold_change_rows
            )
        ),
    )

    #
    # --------------------------------------------------
    # Threshold ID must also remain identical.
    # --------------------------------------------------
    #
    threshold_id_rows = copy.deepcopy(
        rows
    )

    changed = next(
        row
        for row in threshold_id_rows
        if (
            row[
                "condition"
            ]
            == "RSZ"
            and row[
                "analysis_eligible"
            ]
            is True
        )
    )

    changed[
        "threshold_id"
    ] = "different_threshold"

    assert_raises(
        ValueError,
        lambda: (
            build_paired_metric_groups(
                threshold_id_rows
            )
        ),
    )

    #
    # --------------------------------------------------
    # Concrete-video identity must match.
    # --------------------------------------------------
    #
    identity_rows = copy.deepcopy(
        rows
    )

    changed = next(
        row
        for row in identity_rows
        if (
            row[
                "condition"
            ]
            == "RSZ"
            and row[
                "analysis_eligible"
            ]
            is True
        )
    )

    changed[
        "relative_source_path"
    ] = "wrong/path.mp4"

    assert_raises(
        ValueError,
        lambda: (
            build_paired_metric_groups(
                identity_rows
            )
        ),
    )

    #
    # --------------------------------------------------
    # A degraded context without CLN is forbidden.
    # --------------------------------------------------
    #
    only_rsz = [
        copy.deepcopy(
            row
        )
        for row in rows
        if (
            row[
                "condition"
            ]
            == "RSZ"
        )
    ]

    assert_raises(
        ValueError,
        lambda: (
            build_paired_metric_groups(
                only_rsz
            )
        ),
    )

    print(
        "PAIRED METRIC GROUP CONTRACT VALIDATION"
    )
    print(
        "  CLN -> RSZ intersection:       PASSED"
    )
    print(
        "  CLN -> H40 intersection:       PASSED"
    )
    print(
        "  class-specific pairing:        PASSED"
    )
    print(
        "  pairing order preserved:       PASSED"
    )
    print(
        "  condition failure exclusion:   PASSED"
    )
    print(
        "  pooled pairing:                PASSED"
    )
    print(
        "  frozen threshold value:        PASSED"
    )
    print(
        "  frozen threshold identity:     PASSED"
    )
    print(
        "  concrete-video identity:       PASSED"
    )
    print(
        "  degraded-without-CLN rejection:PASSED"
    )
    print()
    print(
        "PAIRED METRIC GROUP CONTRACT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()