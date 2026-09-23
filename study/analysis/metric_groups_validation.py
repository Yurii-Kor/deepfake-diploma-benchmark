from __future__ import annotations

import csv
import copy
from pathlib import Path

from study.analysis.metric_groups import (
    CELEB_DATASET,
    FFPP_DATASET,
    build_primary_metric_groups,
)


REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[
        2
    ]
)

MANIFEST_PATH = (
    REPO_ROOT
    / "study"
    / "manifests"
    / "artifacts"
    / "study_manifest.csv"
)

CHECKPOINT = (
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)


def read_manifest():
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(
            csv.DictReader(
                file
            )
        )


def synthetic_decision_record(
    *,
    manifest_row,
    condition="CLN",
    eligible=True,
):
    label = int(
        manifest_row[
            "study_label"
        ]
    )

    return {
        "schema_version": 1,

        "detector": "xception",
        "checkpoint_sha256": (
            CHECKPOINT
        ),

        "dataset": (
            manifest_row[
                "dataset"
            ]
        ),

        "role": (
            manifest_row[
                "role"
            ]
        ),

        "relative_source_path": (
            manifest_row[
                "relative_source_path"
            ]
        ),

        "base_video_id": (
            manifest_row[
                "base_video_id"
            ]
        ),

        "source_video_id": (
            manifest_row[
                "source_video_id"
            ]
        ),

        "study_label": (
            label
        ),

        "manipulation": (
            manifest_row[
                "manipulation"
            ]
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
            (
                0.10
                if label == 0
                else 0.90
            )
            if eligible
            else None
        ),

        "decision_status": (
            "decided"
            if eligible
            else "not_scored"
        ),

        "decision": (
            label
            if eligible
            else None
        ),
    }


def get_group(
    groups,
    *,
    dataset,
    group_name,
):
    matches = [
        group
        for group in groups
        if (
            group[
                "dataset"
            ]
            == dataset
            and group[
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
    manifest = read_manifest()

    #
    # --------------------------------------------------
    # 1. Build a synthetic but membership-realistic
    #    complete CLN evaluation universe directly
    #    from the frozen canonical manifest.
    # --------------------------------------------------
    #
    evaluation_manifest_rows = [
        row
        for row in manifest
        if (
            (
                row[
                    "dataset"
                ]
                == FFPP_DATASET
                and row[
                    "role"
                ]
                == "test"
            )
            or (
                row[
                    "dataset"
                ]
                == CELEB_DATASET
                and row[
                    "role"
                ]
                == "external_evaluation"
            )
        )
    ]

    assert len(
        evaluation_manifest_rows
    ) == (
        700
        + 518
    )

    decision_rows = [
        synthetic_decision_record(
            manifest_row=row
        )
        for row in (
            evaluation_manifest_rows
        )
    ]

    #
    # Add FF++ validation rows deliberately.
    #
    # They must be accepted as valid analysis records
    # but excluded from primary evaluation groups.
    #
    validation_manifest_rows = [
        row
        for row in manifest
        if (
            row[
                "dataset"
            ]
            == FFPP_DATASET
            and row[
                "role"
            ]
            == "validation"
            and row[
                "study_label"
            ]
            == "0"
        )
    ]

    assert len(
        validation_manifest_rows
    ) == 140

    decision_rows.extend(
        synthetic_decision_record(
            manifest_row=row
        )
        for row in (
            validation_manifest_rows
        )
    )

    groups = (
        build_primary_metric_groups(
            decision_rows
        )
    )

    #
    # One detector × one condition:
    #
    # FF++:
    #   4 method-specific
    #   1 pooled
    #
    # Celeb:
    #   1 pooled
    #
    assert len(
        groups
    ) == 6

    #
    # --------------------------------------------------
    # 2. FF++ method-specific groups.
    # --------------------------------------------------
    #
    for manipulation in (
        "DeepFakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
    ):
        group = get_group(
            groups,
            dataset=(
                FFPP_DATASET
            ),
            group_name=(
                manipulation
            ),
        )

        assert (
            group[
                "group_type"
            ]
            == "manipulation_specific"
        )

        assert (
            group[
                "role"
            ]
            == "test"
        )

        assert (
            group[
                "condition"
            ]
            == "CLN"
        )

        assert (
            group[
                "n_real_nominal"
            ]
            == 140
        )

        assert (
            group[
                "n_manipulated_nominal"
            ]
            == 140
        )

        assert (
            group[
                "n_real_valid"
            ]
            == 140
        )

        assert (
            group[
                "n_manipulated_valid"
            ]
            == 140
        )

        assert {
            row[
                "manipulation"
            ]
            for row in (
                group[
                    "manipulated_records"
                ]
            )
        } == {
            manipulation
        }

        real_paths = [
            row[
                "relative_source_path"
            ]
            for row in (
                group[
                    "real_records"
                ]
            )
        ]

        assert len(
            real_paths
        ) == len(
            set(
                real_paths
            )
        )

    #
    # --------------------------------------------------
    # 3. FF++ pooled group.
    #
    # Real videos must occur exactly once.
    # --------------------------------------------------
    #
    ffpp_pooled = get_group(
        groups,
        dataset=FFPP_DATASET,
        group_name="pooled",
    )

    assert (
        ffpp_pooled[
            "group_type"
        ]
        == "pooled"
    )

    assert (
        ffpp_pooled[
            "n_real_nominal"
        ]
        == 140
    )

    assert (
        ffpp_pooled[
            "n_manipulated_nominal"
        ]
        == 560
    )

    assert (
        ffpp_pooled[
            "n_real_valid"
        ]
        == 140
    )

    assert (
        ffpp_pooled[
            "n_manipulated_valid"
        ]
        == 560
    )

    pooled_real_paths = [
        row[
            "relative_source_path"
        ]
        for row in (
            ffpp_pooled[
                "real_records"
            ]
        )
    ]

    assert len(
        pooled_real_paths
    ) == 140

    assert len(
        set(
            pooled_real_paths
        )
    ) == 140

    pooled_fake_paths = [
        row[
            "relative_source_path"
        ]
        for row in (
            ffpp_pooled[
                "manipulated_records"
            ]
        )
    ]

    assert len(
        pooled_fake_paths
    ) == 560

    assert len(
        set(
            pooled_fake_paths
        )
    ) == 560

    assert {
        row[
            "manipulation"
        ]
        for row in (
            ffpp_pooled[
                "manipulated_records"
            ]
        )
    } == {
        "DeepFakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
    }

    #
    # --------------------------------------------------
    # 4. Explicit regression for the FF++ base-ID
    #    collision discovered before 5.6.
    #
    # The same base_video_id may represent one fake
    # source-target direction in all four methods.
    # They must remain four concrete video records.
    # --------------------------------------------------
    #
    collision = [
        row
        for row in (
            ffpp_pooled[
                "manipulated_records"
            ]
        )
        if (
            row[
                "base_video_id"
            ]
            == "000_003"
        )
    ]

    assert len(
        collision
    ) == 4

    assert len(
        {
            row[
                "relative_source_path"
            ]
            for row in collision
        }
    ) == 4

    assert {
        row[
            "manipulation"
        ]
        for row in collision
    } == {
        "DeepFakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
    }

    #
    # --------------------------------------------------
    # 5. Celeb-DF-v2 pooled external group.
    # --------------------------------------------------
    #
    celeb_pooled = get_group(
        groups,
        dataset=CELEB_DATASET,
        group_name="pooled",
    )

    assert (
        celeb_pooled[
            "role"
        ]
        == "external_evaluation"
    )

    assert (
        celeb_pooled[
            "n_real_nominal"
        ]
        == 178
    )

    assert (
        celeb_pooled[
            "n_manipulated_nominal"
        ]
        == 340
    )

    assert (
        celeb_pooled[
            "n_real_valid"
        ]
        == 178
    )

    assert (
        celeb_pooled[
            "n_manipulated_valid"
        ]
        == 340
    )

    #
    # --------------------------------------------------
    # 6. FF++ validation was excluded from primary
    #    evaluation groups.
    # --------------------------------------------------
    #
    assert all(
        not (
            group[
                "dataset"
            ]
            == FFPP_DATASET
            and group[
                "role"
            ]
            == "validation"
        )
        for group in groups
    )

    #
    # --------------------------------------------------
    # 7. Invalid records remain in nominal population,
    #    but are excluded from valid metric input.
    # --------------------------------------------------
    #
    small_rows = []

    small_manifest = [
        row
        for row in manifest
        if (
            row[
                "dataset"
            ]
            == FFPP_DATASET
            and row[
                "role"
            ]
            == "test"
        )
    ]

    real_rows = [
        row
        for row in small_manifest
        if (
            row[
                "study_label"
            ]
            == "0"
        )
    ][
        :2
    ]

    df_rows = [
        row
        for row in small_manifest
        if (
            row[
                "study_label"
            ]
            == "1"
            and row[
                "manipulation"
            ]
            == "DeepFakes"
        )
    ][
        :2
    ]

    for row in real_rows:
        small_rows.append(
            synthetic_decision_record(
                manifest_row=row
            )
        )

    for row in df_rows:
        small_rows.append(
            synthetic_decision_record(
                manifest_row=row
            )
        )

    #
    # Make one real video invalid.
    #
    small_rows[
        0
    ] = synthetic_decision_record(
        manifest_row=(
            real_rows[
                0
            ]
        ),
        eligible=False,
    )

    small_groups = (
        build_primary_metric_groups(
            small_rows
        )
    )

    small_df_group = get_group(
        small_groups,
        dataset=FFPP_DATASET,
        group_name="DeepFakes",
    )

    assert (
        small_df_group[
            "n_real_nominal"
        ]
        == 2
    )

    assert (
        small_df_group[
            "n_real_valid"
        ]
        == 1
    )

    assert (
        small_df_group[
            "n_real_invalid"
        ]
        == 1
    )

    assert (
        small_df_group[
            "n_manipulated_nominal"
        ]
        == 2
    )

    assert (
        small_df_group[
            "n_manipulated_valid"
        ]
        == 2
    )

    #
    # --------------------------------------------------
    # 8. Duplicate concrete decision record rejected.
    # --------------------------------------------------
    #
    duplicate_rows = [
        decision_rows[
            0
        ],
        copy.deepcopy(
            decision_rows[
                0
            ]
        ),
    ]

    assert_raises(
        ValueError,
        lambda: (
            build_primary_metric_groups(
                duplicate_rows
            )
        ),
    )

    #
    # --------------------------------------------------
    # 9. Multiple checkpoints for one detector rejected.
    # --------------------------------------------------
    #
    checkpoint_rows = [
        copy.deepcopy(
            decision_rows[
                0
            ]
        ),
        copy.deepcopy(
            decision_rows[
                1
            ]
        ),
    ]

    checkpoint_rows[
        1
    ][
        "checkpoint_sha256"
    ] = (
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    )

    assert_raises(
        ValueError,
        lambda: (
            build_primary_metric_groups(
                checkpoint_rows
            )
        ),
    )

    #
    # --------------------------------------------------
    # 10. Invalid FF++ manipulation rejected.
    # --------------------------------------------------
    #
    bad_manipulation = next(
        copy.deepcopy(
            row
        )
        for row in decision_rows
        if (
            row[
                "dataset"
            ]
            == FFPP_DATASET
            and row[
                "study_label"
            ]
            == 1
        )
    )

    bad_manipulation[
        "manipulation"
    ] = "UNKNOWN"

    assert_raises(
        ValueError,
        lambda: (
            build_primary_metric_groups(
                [
                    bad_manipulation
                ]
            )
        ),
    )

    print(
        "METRIC GROUP CONTRACT VALIDATION"
    )
    print(
        "  canonical evaluation videos:  1218"
    )
    print(
        "  FF++ primary groups:          5"
    )
    print(
        "  Celeb primary groups:         1"
    )
    print(
        "  FF++ method real count:       140"
    )
    print(
        "  FF++ method fake count:       140"
    )
    print(
        "  FF++ pooled real count:       140"
    )
    print(
        "  FF++ pooled fake count:       560"
    )
    print(
        "  Celeb real count:             178"
    )
    print(
        "  Celeb fake count:             340"
    )
    print(
        "  FF++ real deduplication:      PASSED"
    )
    print(
        "  FF++ base-ID collision:       PASSED"
    )
    print(
        "  validation exclusion:         PASSED"
    )
    print(
        "  invalid-record accounting:    PASSED"
    )
    print(
        "  duplicate rejection:          PASSED"
    )
    print(
        "  checkpoint consistency:       PASSED"
    )
    print(
        "  manipulation validation:      PASSED"
    )
    print()
    print(
        "METRIC GROUP CONTRACT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()