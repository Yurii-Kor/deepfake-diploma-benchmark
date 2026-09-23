from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from study.analysis.analysis_schema import (
    ANALYSIS_STATUS_INVALID,
    ANALYSIS_STATUS_VALID,
    analysis_key,
    build_analysis_record,
)
from study.analysis.build_analysis_records import (
    build_manifest_index,
    build_records,
    read_csv,
    read_jsonl,
)


REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[
        2
    ]
)

STUDY_DATA_ROOT = (
    Path.home()
    / "deepfake_lab"
    / "study_data"
)

MANIFEST_PATH = (
    REPO_ROOT
    / "study"
    / "manifests"
    / "artifacts"
    / "study_manifest.csv"
)

INFERENCE_SMOKE_PATH = (
    STUDY_DATA_ROOT
    / "evaluation"
    / "canonical_inference_smoke.jsonl"
)


def synthetic_inference_row(
    *,
    manifest_row,
    role,
    condition="CLN",
    inference_status="ok",
    video_score=0.5,
):
    if inference_status == "ok":
        aggregation_method = (
            "unweighted_arithmetic_mean"
        )

        successful_frame_count = 32

    else:
        aggregation_method = None
        successful_frame_count = 0

    return {
        "detector": "xception",
        "checkpoint_sha256": (
            "synthetic-contract-checkpoint"
        ),

        "dataset": (
            manifest_row[
                "dataset"
            ]
        ),
        "role": role,

        "base_video_id": (
            manifest_row[
                "base_video_id"
            ]
        ),
        "relative_source_path": (
            manifest_row[
                "relative_source_path"
            ]
        ),

        "study_label": int(
            manifest_row[
                "study_label"
            ]
        ),

        "condition": (
            condition
        ),

        "target_frame_budget": 32,
        "clean_valid_frame_count": 32,
        "successful_frame_count": (
            successful_frame_count
        ),

        "aggregation_method": (
            aggregation_method
        ),

        "video_score": (
            video_score
        ),

        "inference_status": (
            inference_status
        ),

        "failure_stage": (
            ""
            if inference_status == "ok"
            else "synthetic_failure"
        ),

        "failure_reason": (
            ""
            if inference_status == "ok"
            else "synthetic contract failure"
        ),
    }


def main():
    manifest_rows = read_csv(
        MANIFEST_PATH
    )

    manifest_index = (
        build_manifest_index(
            manifest_rows
        )
    )

    inference_rows = read_jsonl(
        INFERENCE_SMOKE_PATH
    )

    analysis_rows = (
        build_records(
            inference_rows=(
                inference_rows
            ),
            manifest_index=(
                manifest_index
            ),
        )
    )

    #
    # --------------------------------------------------
    # 1. Real 5.4 smoke joins exactly into 5.6.1.
    # --------------------------------------------------
    #
    assert len(
        inference_rows
    ) == 18

    assert len(
        analysis_rows
    ) == 18

    assert len(
        {
            analysis_key(
                row
            )
            for row in analysis_rows
        }
    ) == 18

    assert all(
        row[
            "analysis_status"
        ]
        == ANALYSIS_STATUS_VALID
        for row in analysis_rows
    )

    assert all(
        row[
            "analysis_eligible"
        ]
        is True
        for row in analysis_rows
    )

    assert all(
        row[
            "dataset"
        ]
        == "FaceForensics++"
        for row in analysis_rows
    )

    assert all(
        row[
            "role"
        ]
        == "validation"
        for row in analysis_rows
    )

    assert all(
        row[
            "source_split"
        ]
        == "val"
        for row in analysis_rows
    )

    assert all(
        row[
            "study_label"
        ]
        == 0
        for row in analysis_rows
    )

    assert all(
        row[
            "manipulation"
        ]
        == ""
        for row in analysis_rows
    )

    assert all(
        row[
            "bootstrap_stratum"
        ]
        == "real"
        for row in analysis_rows
    )

    assert all(
        row[
            "bootstrap_unit_id"
        ]
        == row[
            "source_video_id"
        ]
        for row in analysis_rows
    )

    assert len(
        {
            row[
                "video_uid"
            ]
            for row in analysis_rows
        }
    ) == 3

    #
    # Scores must be unchanged by the analysis adapter.
    #
    inference_scores = {
        (
            row[
                "detector"
            ],
            row[
                "dataset"
            ],
            row[
                "relative_source_path"
            ],
            row[
                "condition"
            ],
        ): float(
            row[
                "video_score"
            ]
        )
        for row in inference_rows
    }

    for row in analysis_rows:
        key = (
            row[
                "detector"
            ],
            row[
                "dataset"
            ],
            row[
                "relative_source_path"
            ],
            row[
                "condition"
            ],
        )

        assert (
            row[
                "video_score"
            ]
            == inference_scores[
                key
            ]
        )

    #
    # --------------------------------------------------
    # 2. Real FF++ manifest collision:
    #    same base_video_id across manipulation methods
    #    must remain four distinct videos.
    # --------------------------------------------------
    #
    collision_rows = [
        row
        for row in manifest_rows
        if (
            row[
                "dataset"
            ]
            == "FaceForensics++"
            and row[
                "role"
            ]
            == "test"
            and row[
                "study_label"
            ]
            == "1"
            and row[
                "base_video_id"
            ]
            == "000_003"
        )
    ]

    assert len(
        collision_rows
    ) == 4

    assert {
        row[
            "manipulation"
        ]
        for row in collision_rows
    } == {
        "DeepFakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
    }

    synthetic_ffpp = [
        build_analysis_record(
            inference_row=(
                synthetic_inference_row(
                    manifest_row=row,
                    role="test",
                )
            ),
            manifest_row=row,
        )
        for row in collision_rows
    ]

    assert len(
        {
            row[
                "video_uid"
            ]
            for row in synthetic_ffpp
        }
    ) == 4

    assert len(
        {
            analysis_key(
                row
            )
            for row in synthetic_ffpp
        }
    ) == 4

    assert len(
        {
            row[
                "base_video_id"
            ]
            for row in synthetic_ffpp
        }
    ) == 1

    assert len(
        {
            row[
                "bootstrap_unit_id"
            ]
            for row in synthetic_ffpp
        }
    ) == 1

    assert {
        row[
            "bootstrap_stratum"
        ]
        for row in synthetic_ffpp
    } == {
        "DeepFakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
    }

    #
    # --------------------------------------------------
    # 3. Celeb execution/test role is normalized to
    #    authoritative external_evaluation role.
    # --------------------------------------------------
    #
    celeb_real = next(
        row
        for row in manifest_rows
        if (
            row[
                "dataset"
            ]
            == "Celeb-DF-v2"
            and row[
                "role"
            ]
            == "external_evaluation"
            and row[
                "study_label"
            ]
            == "0"
        )
    )

    celeb_analysis = (
        build_analysis_record(
            inference_row=(
                synthetic_inference_row(
                    manifest_row=(
                        celeb_real
                    ),
                    role="test",
                )
            ),
            manifest_row=(
                celeb_real
            ),
        )
    )

    assert (
        celeb_analysis[
            "inference_role"
        ]
        == "test"
    )

    assert (
        celeb_analysis[
            "role"
        ]
        == "external_evaluation"
    )

    assert (
        celeb_analysis[
            "bootstrap_stratum"
        ]
        == "real"
    )

    assert (
        celeb_analysis[
            "bootstrap_unit_id"
        ]
        == celeb_analysis[
            "base_video_id"
        ]
    )

    #
    # --------------------------------------------------
    # 4. Invalid inference remains represented but is
    #    excluded from score-based analysis.
    # --------------------------------------------------
    #
    invalid_record = (
        build_analysis_record(
            inference_row=(
                synthetic_inference_row(
                    manifest_row=(
                        celeb_real
                    ),
                    role="test",
                    condition="RSZ",
                    inference_status=(
                        "invalid_condition"
                    ),
                    video_score=None,
                )
            ),
            manifest_row=(
                celeb_real
            ),
        )
    )

    assert (
        invalid_record[
            "analysis_status"
        ]
        == ANALYSIS_STATUS_INVALID
    )

    assert (
        invalid_record[
            "analysis_eligible"
        ]
        is False
    )

    assert (
        invalid_record[
            "video_score"
        ]
        is None
    )

    #
    # --------------------------------------------------
    # 5. Duplicate detector-video-condition records
    #    must fail instead of silently overwriting.
    # --------------------------------------------------
    #
    duplicate_test_passed = False

    try:
        build_records(
            inference_rows=[
                inference_rows[
                    0
                ],
                inference_rows[
                    0
                ],
            ],
            manifest_index=(
                manifest_index
            ),
        )

    except ValueError:
        duplicate_test_passed = True

    assert duplicate_test_passed

    detector_counts = Counter(
        row[
            "detector"
        ]
        for row in analysis_rows
    )

    condition_counts = Counter(
        row[
            "condition"
        ]
        for row in analysis_rows
    )

    rows_per_video = defaultdict(
        int
    )

    for row in analysis_rows:
        rows_per_video[
            row[
                "video_uid"
            ]
        ] += 1

    assert set(
        rows_per_video.values()
    ) == {
        6
    }

    print(
        "ANALYSIS RECORD CONTRACT VALIDATION"
    )
    print(
        "  real 5.4 smoke records:       18/18 PASSED"
    )
    print(
        "  canonical manifest join:       PASSED"
    )
    print(
        "  detector counts:              {}".format(
            dict(
                sorted(
                    detector_counts.items()
                )
            )
        )
    )
    print(
        "  condition counts:             {}".format(
            dict(
                sorted(
                    condition_counts.items()
                )
            )
        )
    )
    print(
        "  score preservation:           PASSED"
    )
    print(
        "  FF++ base-ID collision:       PASSED"
    )
    print(
        "  FF++ bootstrap source unit:   PASSED"
    )
    print(
        "  FF++ manipulation strata:     PASSED"
    )
    print(
        "  Celeb role normalization:     PASSED"
    )
    print(
        "  Celeb bootstrap base unit:    PASSED"
    )
    print(
        "  invalid record preservation:  PASSED"
    )
    print(
        "  duplicate rejection:          PASSED"
    )
    print()
    print(
        "ANALYSIS RECORD CONTRACT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()