from __future__ import annotations

from copy import deepcopy

from study.reproducibility.validate_inference_analysis_chain import (
    validate_inference_analysis_chain,
    validate_inference_record,
)


CHECKPOINT = (
    "0123456789abcdef"
    "0123456789abcdef"
    "0123456789abcdef"
    "0123456789abcdef"
)


def expect_value_error(
    function,
    *,
    description: str,
):
    try:
        function()
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError: {}".format(
            description
        )
    )


def valid_inference_record():
    return {
        "detector": "xception",
        "checkpoint_sha256": CHECKPOINT,
        "dataset": "FaceForensics++",
        "role": "test",
        "subgroup": "original",
        "study_label": 0,
        "source_label": "real",
        "base_video_id": "001",
        "relative_source_path": (
            "original/test/001.mp4"
        ),
        "condition": "CLN",
        "target_frame_budget": 32,
        "clean_valid_frame_count": 3,
        "retained_temporal_positions": [
            0,
            10,
            31,
        ],
        "successful_frame_count": 3,
        "aggregation_method": (
            "unweighted_arithmetic_mean"
        ),
        "video_score": 0.25,
        "inference_status": "ok",
        "failure_stage": "",
        "failure_reason": "",
    }


def valid_analysis_record():
    return {
        "schema_version": 1,

        "detector": "xception",
        "checkpoint_sha256": CHECKPOINT,

        "dataset": "FaceForensics++",
        "source_split": "test",
        "role": "test",
        "inference_role": "test",

        "video_uid": (
            "FaceForensics++::"
            "original/test/001.mp4"
        ),
        "pairing_uid": (
            "FaceForensics++::"
            "original/test/001.mp4"
        ),

        "base_video_id": "001",
        "source_group_id": "001",
        "source_video_id": "001",
        "target_video_id": "",

        "relative_source_path": (
            "original/test/001.mp4"
        ),

        "study_label": 0,
        "manipulation": "",
        "source_label": "real",

        "bootstrap_stratum": "real",
        "bootstrap_unit_id": "001",
        "bootstrap_cluster_uid": (
            "FaceForensics++::test::001"
        ),

        "condition": "CLN",

        "target_frame_budget": 32,
        "clean_valid_frame_count": 3,
        "successful_frame_count": 3,

        "complete_target_frame_set": False,

        "aggregation_method": (
            "unweighted_arithmetic_mean"
        ),

        "video_score": 0.25,

        "inference_status": "ok",
        "analysis_status": "valid_score",
        "analysis_eligible": True,

        "failure_stage": "",
        "failure_reason": "",
    }


def main():
    inference = valid_inference_record()

    validate_inference_record(
        inference
    )

    analysis = valid_analysis_record()

    validate_inference_analysis_chain(
        inference_records=[
            inference
        ],
        analysis_records=[
            analysis
        ],
    )

    float_label = deepcopy(
        inference
    )

    float_label[
        "study_label"
    ] = 0.7

    expect_value_error(
        lambda: validate_inference_record(
            float_label
        ),
        description=(
            "float inference study_label"
        ),
    )

    boolean_label = deepcopy(
        inference
    )

    boolean_label[
        "study_label"
    ] = False

    expect_value_error(
        lambda: validate_inference_record(
            boolean_label
        ),
        description=(
            "boolean inference study_label"
        ),
    )

    boolean_count = deepcopy(
        inference
    )

    boolean_count[
        "successful_frame_count"
    ] = True

    expect_value_error(
        lambda: validate_inference_record(
            boolean_count
        ),
        description=(
            "boolean frame count"
        ),
    )

    duplicate_positions = deepcopy(
        inference
    )

    duplicate_positions[
        "retained_temporal_positions"
    ] = [
        0,
        10,
        10,
    ]

    expect_value_error(
        lambda: validate_inference_record(
            duplicate_positions
        ),
        description=(
            "duplicate temporal positions"
        ),
    )

    unsorted_positions = deepcopy(
        inference
    )

    unsorted_positions[
        "retained_temporal_positions"
    ] = [
        10,
        0,
        31,
    ]

    expect_value_error(
        lambda: validate_inference_record(
            unsorted_positions
        ),
        description=(
            "unsorted temporal positions"
        ),
    )

    invalid_state = deepcopy(
        inference
    )

    invalid_state[
        "inference_status"
    ] = "invalid_condition"

    invalid_state[
        "video_score"
    ] = None

    invalid_state[
        "aggregation_method"
    ] = None

    invalid_state[
        "successful_frame_count"
    ] = 2

    invalid_state[
        "failure_stage"
    ] = "detector_inference"

    invalid_state[
        "failure_reason"
    ] = "synthetic failure"

    validate_inference_record(
        invalid_state
    )

    changed_label = deepcopy(
        analysis
    )

    changed_label[
        "study_label"
    ] = 1

    expect_value_error(
        lambda: validate_inference_analysis_chain(
            inference_records=[
                inference
            ],
            analysis_records=[
                changed_label
            ],
        ),
        description=(
            "analysis changed study_label"
        ),
    )

    changed_score = deepcopy(
        analysis
    )

    changed_score[
        "video_score"
    ] = 0.5

    expect_value_error(
        lambda: validate_inference_analysis_chain(
            inference_records=[
                inference
            ],
            analysis_records=[
                changed_score
            ],
        ),
        description=(
            "analysis changed video_score"
        ),
    )

    changed_checkpoint = deepcopy(
        analysis
    )

    changed_checkpoint[
        "checkpoint_sha256"
    ] = (
        "abcdef0123456789"
        "abcdef0123456789"
        "abcdef0123456789"
        "abcdef0123456789"
    )

    expect_value_error(
        lambda: validate_inference_analysis_chain(
            inference_records=[
                inference
            ],
            analysis_records=[
                changed_checkpoint
            ],
        ),
        description=(
            "analysis changed checkpoint"
        ),
    )

    changed_validity = deepcopy(
        analysis
    )

    changed_validity[
        "analysis_eligible"
    ] = False

    expect_value_error(
        lambda: validate_inference_analysis_chain(
            inference_records=[
                inference
            ],
            analysis_records=[
                changed_validity
            ],
        ),
        description=(
            "analysis changed validity"
        ),
    )

    print(
        "INFERENCE -> ANALYSIS CONTRACT VALIDATION"
    )
    print(
        "  exact binary inference label:    PASSED"
    )
    print(
        "  float inference label rejected:  PASSED"
    )
    print(
        "  boolean inference label rejected:PASSED"
    )
    print(
        "  strict frame counts:             PASSED"
    )
    print(
        "  duplicate positions rejected:    PASSED"
    )
    print(
        "  unsorted positions rejected:     PASSED"
    )
    print(
        "  valid/invalid state schema:       PASSED"
    )
    print(
        "  changed label detected:           PASSED"
    )
    print(
        "  changed score detected:           PASSED"
    )
    print(
        "  changed checkpoint detected:      PASSED"
    )
    print(
        "  changed validity detected:        PASSED"
    )
    print()
    print(
        "INFERENCE -> ANALYSIS CONTRACT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()