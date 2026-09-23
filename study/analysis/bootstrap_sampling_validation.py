from __future__ import annotations

from study.analysis.bootstrap_sampling import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    bootstrap_sampling_key,
    bootstrap_stratum,
    bootstrap_unit_id,
    build_stratum_units,
    sample_bootstrap_replicate,
)


def ffpp_row(
    *,
    source_video_id,
    label,
    manipulation="",
):
    return {
        "dataset": (
            "FaceForensics++"
        ),

        "study_label": (
            label
        ),

        "source_video_id": (
            source_video_id
        ),

        "base_video_id": (
            source_video_id
        ),

        "manipulation": (
            manipulation
        ),
    }


def celeb_row(
    *,
    base_video_id,
    label,
):
    return {
        "dataset": (
            "Celeb-DF-v2"
        ),

        "study_label": (
            label
        ),

        "source_video_id": (
            base_video_id
        ),

        "base_video_id": (
            base_video_id
        ),

        "manipulation": "",
    }


def main():
    assert (
        BOOTSTRAP_REPLICATES
        == 2000
    )

    assert (
        BOOTSTRAP_CI_LEVEL
        == 0.95
    )

    assert (
        BOOTSTRAP_SEED
        == 1024
    )

    #
    # --------------------------------------------------
    # 1. FF++ unit contract.
    # --------------------------------------------------
    #
    real = ffpp_row(
        source_video_id="000",
        label=0,
    )

    df = ffpp_row(
        source_video_id="000",
        label=1,
        manipulation="DeepFakes",
    )

    f2f = ffpp_row(
        source_video_id="000",
        label=1,
        manipulation="Face2Face",
    )

    assert (
        bootstrap_unit_id(
            real
        )
        == "000"
    )

    assert (
        bootstrap_unit_id(
            df
        )
        == "000"
    )

    assert (
        bootstrap_stratum(
            real
        )
        == "real"
    )

    assert (
        bootstrap_stratum(
            df
        )
        == "DeepFakes"
    )

    assert (
        bootstrap_stratum(
            f2f
        )
        == "Face2Face"
    )

    #
    # Same raw source ID in different strata must
    # remain distinct sampling units.
    #
    assert (
        bootstrap_sampling_key(
            real
        )
        != bootstrap_sampling_key(
            df
        )
    )

    assert (
        bootstrap_sampling_key(
            df
        )
        != bootstrap_sampling_key(
            f2f
        )
    )

    #
    # --------------------------------------------------
    # 2. FF++ stratification.
    # --------------------------------------------------
    #
    ffpp_rows = [
        ffpp_row(
            source_video_id="001",
            label=0,
        ),
        ffpp_row(
            source_video_id="002",
            label=0,
        ),

        ffpp_row(
            source_video_id="001",
            label=1,
            manipulation="DeepFakes",
        ),
        ffpp_row(
            source_video_id="002",
            label=1,
            manipulation="DeepFakes",
        ),

        ffpp_row(
            source_video_id="001",
            label=1,
            manipulation="Face2Face",
        ),
        ffpp_row(
            source_video_id="002",
            label=1,
            manipulation="Face2Face",
        ),

        ffpp_row(
            source_video_id="001",
            label=1,
            manipulation="FaceSwap",
        ),
        ffpp_row(
            source_video_id="002",
            label=1,
            manipulation="FaceSwap",
        ),

        ffpp_row(
            source_video_id="001",
            label=1,
            manipulation="NeuralTextures",
        ),
        ffpp_row(
            source_video_id="002",
            label=1,
            manipulation="NeuralTextures",
        ),
    ]

    ffpp_units = build_stratum_units(
        ffpp_rows
    )

    assert (
        set(
            ffpp_units
        )
        == {
            "real",
            "DeepFakes",
            "Face2Face",
            "FaceSwap",
            "NeuralTextures",
        }
    )

    for units in (
        ffpp_units.values()
    ):
        assert units == [
            "001",
            "002",
        ]

    #
    # --------------------------------------------------
    # 3. Sampling preserves each stratum size.
    # --------------------------------------------------
    #
    sampled = sample_bootstrap_replicate(
        ffpp_rows,
        replicate_index=0,
    )

    for stratum in (
        ffpp_units
    ):
        assert (
            len(
                sampled[
                    stratum
                ]
            )
            == len(
                ffpp_units[
                    stratum
                ]
            )
        )

        assert set(
            sampled[
                stratum
            ]
        ).issubset(
            set(
                ffpp_units[
                    stratum
                ]
            )
        )

    #
    # --------------------------------------------------
    # 4. Same seed + replicate => byte-for-byte same
    #    sampled identifiers.
    # --------------------------------------------------
    #
    sampled_again = (
        sample_bootstrap_replicate(
            ffpp_rows,
            replicate_index=0,
        )
    )

    assert (
        sampled
        == sampled_again
    )

    #
    # --------------------------------------------------
    # 5. Replicate streams are independently
    #    addressable.
    # --------------------------------------------------
    #
    replicate_1_a = (
        sample_bootstrap_replicate(
            ffpp_rows,
            replicate_index=1,
        )
    )

    replicate_1_b = (
        sample_bootstrap_replicate(
            ffpp_rows,
            replicate_index=1,
        )
    )

    assert (
        replicate_1_a
        == replicate_1_b
    )

    #
    # --------------------------------------------------
    # 6. Celeb uses base_video_id and two strata.
    # --------------------------------------------------
    #
    celeb_rows = [
        celeb_row(
            base_video_id="real_1",
            label=0,
        ),
        celeb_row(
            base_video_id="real_2",
            label=0,
        ),
        celeb_row(
            base_video_id="fake_1",
            label=1,
        ),
        celeb_row(
            base_video_id="fake_2",
            label=1,
        ),
        celeb_row(
            base_video_id="fake_3",
            label=1,
        ),
    ]

    celeb_units = build_stratum_units(
        celeb_rows
    )

    assert (
        set(
            celeb_units
        )
        == {
            "real",
            "manipulated",
        }
    )

    assert (
        celeb_units[
            "real"
        ]
        == [
            "real_1",
            "real_2",
        ]
    )

    assert (
        celeb_units[
            "manipulated"
        ]
        == [
            "fake_1",
            "fake_2",
            "fake_3",
        ]
    )

    celeb_sample = (
        sample_bootstrap_replicate(
            celeb_rows,
            replicate_index=0,
        )
    )

    assert (
        len(
            celeb_sample[
                "real"
            ]
        )
        == 2
    )

    assert (
        len(
            celeb_sample[
                "manipulated"
            ]
        )
        == 3
    )

    print(
        "BOOTSTRAP SAMPLING CONTRACT VALIDATION"
    )
    print(
        "  bootstrap replicates = 2000:   PASSED"
    )
    print(
        "  percentile CI = 95%:           PASSED"
    )
    print(
        "  bootstrap seed = 1024:         PASSED"
    )
    print(
        "  FF++ source-video unit:        PASSED"
    )
    print(
        "  FF++ real stratum:             PASSED"
    )
    print(
        "  FF++ method strata:            PASSED"
    )
    print(
        "  same ID across strata safe:    PASSED"
    )
    print(
        "  stratum sizes preserved:       PASSED"
    )
    print(
        "  deterministic replicate RNG:   PASSED"
    )
    print(
        "  Celeb base-video unit:         PASSED"
    )
    print(
        "  Celeb real/fake strata:        PASSED"
    )
    print()
    print(
        "BOOTSTRAP SAMPLING CONTRACT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()