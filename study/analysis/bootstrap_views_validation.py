from __future__ import annotations

from study.analysis.bootstrap_views import (
    build_ffpp_absolute_replicate_plan,
    build_ffpp_paired_replicate_plan,
    materialize_ffpp_absolute_group,
    materialize_ffpp_paired_group,
)
from study.analysis.metric_groups import (
    build_primary_metric_groups,
)
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

THRESHOLD = 0.5

MANIPULATIONS = (
    "DeepFakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
)


def row(
    *,
    source_id,
    condition,
    label,
    manipulation="",
):
    if label == 0:
        path = (
            "original/{}.mp4"
            .format(
                source_id
            )
        )

        score = 0.1

    else:
        path = (
            "{}/{}_target.mp4"
            .format(
                manipulation,
                source_id,
            )
        )

        score = 0.9

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
            "FaceForensics++::{}"
            .format(
                path
            )
        ),

        "relative_source_path": (
            path
        ),

        "base_video_id": (
            source_id
        ),

        "source_video_id": (
            source_id
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

        "analysis_eligible": True,

        "analysis_status": (
            "valid_score"
        ),

        "video_score": (
            score
        ),

        "threshold_id": (
            THRESHOLD_ID
        ),

        "threshold_value": (
            THRESHOLD
        ),

        "decision_status": (
            "decided"
        ),

        "decision": (
            1
            if score >= THRESHOLD
            else 0
        ),
    }


def group_index(
    groups,
):
    return {
        group[
            "group_name"
        ]: group
        for group in groups
    }


def unit_sequence(
    rows,
):
    return [
        row[
            "source_video_id"
        ]
        for row in rows
    ]


def main():
    records = []

    #
    # Four real source units.
    #
    for condition in (
        "CLN",
        "RSZ",
    ):
        for source_id in (
            "001",
            "002",
            "003",
            "004",
        ):
            records.append(
                row(
                    source_id=(
                        source_id
                    ),
                    condition=(
                        condition
                    ),
                    label=0,
                )
            )

        #
        # Four fake units per manipulation method.
        #
        for manipulation in (
            MANIPULATIONS
        ):
            for source_id in (
                "001",
                "002",
                "003",
                "004",
            ):
                records.append(
                    row(
                        source_id=(
                            source_id
                        ),
                        condition=(
                            condition
                        ),
                        label=1,
                        manipulation=(
                            manipulation
                        ),
                    )
                )

    #
    # --------------------------------------------------
    # 1. Absolute CLN groups.
    # --------------------------------------------------
    #
    absolute_groups = (
        build_primary_metric_groups(
            records
        )
    )

    cln_groups = [
        group
        for group in absolute_groups
        if (
            group[
                "condition"
            ]
            == "CLN"
        )
    ]

    assert len(
        cln_groups
    ) == 5

    cln_index = group_index(
        cln_groups
    )

    plan = (
        build_ffpp_absolute_replicate_plan(
            cln_groups,
            replicate_index=0,
            seed=1024,
            stream_name=(
                "validation_shared_absolute"
            ),
        )
    )

    plan_again = (
        build_ffpp_absolute_replicate_plan(
            cln_groups,
            replicate_index=0,
            seed=1024,
            stream_name=(
                "validation_shared_absolute"
            ),
        )
    )

    assert (
        plan
        == plan_again
    )

    #
    # --------------------------------------------------
    # 2. Every method-specific view must receive
    #    the exact same sampled REAL sequence.
    # --------------------------------------------------
    #
    sampled_real_sequences = []

    method_fake_sequences = {}

    for manipulation in (
        MANIPULATIONS
    ):
        (
            sampled_real,
            sampled_fake,
        ) = materialize_ffpp_absolute_group(
            group=(
                cln_index[
                    manipulation
                ]
            ),
            replicate_plan=(
                plan
            ),
        )

        sampled_real_sequences.append(
            unit_sequence(
                sampled_real
            )
        )

        method_fake_sequences[
            manipulation
        ] = unit_sequence(
            sampled_fake
        )

    (
        pooled_real,
        pooled_fake,
    ) = materialize_ffpp_absolute_group(
        group=(
            cln_index[
                "pooled"
            ]
        ),
        replicate_plan=(
            plan
        ),
    )

    sampled_real_sequences.append(
        unit_sequence(
            pooled_real
        )
    )

    first_real_sequence = (
        sampled_real_sequences[
            0
        ]
    )

    assert all(
        sequence
        == first_real_sequence
        for sequence in (
            sampled_real_sequences
        )
    )

    #
    # --------------------------------------------------
    # 3. Pooled fake sample must be EXACTLY the
    #    concatenation of already sampled method strata.
    # --------------------------------------------------
    #
    expected_pooled_fake = []

    for manipulation in (
        MANIPULATIONS
    ):
        expected_pooled_fake.extend(
            method_fake_sequences[
                manipulation
            ]
        )

    assert (
        unit_sequence(
            pooled_fake
        )
        == expected_pooled_fake
    )

    #
    # --------------------------------------------------
    # 4. Paired CLN -> RSZ groups.
    # --------------------------------------------------
    #
    paired_groups = (
        build_paired_metric_groups(
            records
        )
    )

    rsz_groups = [
        group
        for group in paired_groups
        if (
            group[
                "degraded_condition"
            ]
            == "RSZ"
        )
    ]

    assert len(
        rsz_groups
    ) == 5

    rsz_index = group_index(
        rsz_groups
    )

    paired_plan = (
        build_ffpp_paired_replicate_plan(
            rsz_groups,
            replicate_index=0,
            seed=1024,
            stream_name=(
                "validation_shared_paired"
            ),
        )
    )

    #
    # --------------------------------------------------
    # 5. Same sampled IDs must occur on CLN and RSZ
    #    sides of every paired method-specific view.
    # --------------------------------------------------
    #
    paired_real_sequences = []

    paired_method_fake_sequences = {}

    for manipulation in (
        MANIPULATIONS
    ):
        sampled = (
            materialize_ffpp_paired_group(
                group=(
                    rsz_index[
                        manipulation
                    ]
                ),
                replicate_plan=(
                    paired_plan
                ),
            )
        )

        assert (
            unit_sequence(
                sampled[
                    "clean_real"
                ]
            )
            == unit_sequence(
                sampled[
                    "degraded_real"
                ]
            )
        )

        assert (
            unit_sequence(
                sampled[
                    "clean_manipulated"
                ]
            )
            == unit_sequence(
                sampled[
                    "degraded_manipulated"
                ]
            )
        )

        paired_real_sequences.append(
            unit_sequence(
                sampled[
                    "clean_real"
                ]
            )
        )

        paired_method_fake_sequences[
            manipulation
        ] = unit_sequence(
            sampled[
                "clean_manipulated"
            ]
        )

    #
    # --------------------------------------------------
    # 6. Paired pooled view uses the same shared
    #    real draw and the same four fake draws.
    # --------------------------------------------------
    #
    pooled_paired = (
        materialize_ffpp_paired_group(
            group=(
                rsz_index[
                    "pooled"
                ]
            ),
            replicate_plan=(
                paired_plan
            ),
        )
    )

    assert (
        unit_sequence(
            pooled_paired[
                "clean_real"
            ]
        )
        == unit_sequence(
            pooled_paired[
                "degraded_real"
            ]
        )
    )

    assert (
        unit_sequence(
            pooled_paired[
                "clean_manipulated"
            ]
        )
        == unit_sequence(
            pooled_paired[
                "degraded_manipulated"
            ]
        )
    )

    paired_real_sequences.append(
        unit_sequence(
            pooled_paired[
                "clean_real"
            ]
        )
    )

    first_paired_real = (
        paired_real_sequences[
            0
        ]
    )

    assert all(
        sequence
        == first_paired_real
        for sequence in (
            paired_real_sequences
        )
    )

    expected_paired_pooled_fake = []

    for manipulation in (
        MANIPULATIONS
    ):
        expected_paired_pooled_fake.extend(
            paired_method_fake_sequences[
                manipulation
            ]
        )

    assert (
        unit_sequence(
            pooled_paired[
                "clean_manipulated"
            ]
        )
        == expected_paired_pooled_fake
    )

    #
    # --------------------------------------------------
    # 7. Plan preserves original stratum sizes.
    # --------------------------------------------------
    #
    assert (
        len(
            plan[
                "real"
            ]
        )
        == 4
    )

    for manipulation in (
        MANIPULATIONS
    ):
        assert (
            len(
                plan[
                    manipulation
                ]
            )
            == 4
        )

    print(
        "BOOTSTRAP VIEW CONTRACT VALIDATION"
    )
    print(
        "  deterministic shared plan:     PASSED"
    )
    print(
        "  unified FF++ real resample:    PASSED"
    )
    print(
        "  method-specific fake strata:   PASSED"
    )
    print(
        "  pooled = method resamples:     PASSED"
    )
    print(
        "  paired CLN/RSZ unit identity:  PASSED"
    )
    print(
        "  paired shared real resample:   PASSED"
    )
    print(
        "  paired pooled composition:     PASSED"
    )
    print(
        "  original stratum sizes kept:   PASSED"
    )
    print()
    print(
        "BOOTSTRAP VIEW CONTRACT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()