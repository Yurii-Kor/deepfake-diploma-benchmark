from __future__ import annotations

from copy import deepcopy

from study.reproducibility.validate_manifest_analysis_chain import (
    build_manifest_index,
    strict_csv_binary_int,
    validate_manifest_analysis_chain,
    validate_manifest_record,
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


def ffpp_manifest_record():
    return {
        "dataset": "FaceForensics++",
        "source_split": "test",
        "role": "test",
        "base_video_id": "001",
        "source_group_id": "001_002",
        "source_video_id": "001",
        "target_video_id": "",
        "study_label": "0",
        "manipulation": "",
        "relative_source_path": (
            "original/001.mp4"
        ),
        "source_label": "",
    }


def ffpp_inference_record():
    return {
        "detector": "xception",
        "checkpoint_sha256": CHECKPOINT,
        "dataset": "FaceForensics++",
        "role": "test",
        "subgroup": "original",
        "study_label": 0,
        "source_label": "",
        "base_video_id": "001",
        "relative_source_path": (
            "original/001.mp4"
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


def ffpp_analysis_record():
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
            "original/001.mp4"
        ),
        "pairing_uid": (
            "FaceForensics++::"
            "original/001.mp4"
        ),

        "base_video_id": "001",
        "source_group_id": "001_002",
        "source_video_id": "001",
        "target_video_id": "",

        "relative_source_path": (
            "original/001.mp4"
        ),

        "study_label": 0,
        "manipulation": "",
        "source_label": "",

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


def celeb_manifest_record():
    return {
        "dataset": "Celeb-DF-v2",
        "source_split": "test",
        "role": "external_evaluation",
        "base_video_id": "00011",
        "source_group_id": "00011",
        "source_video_id": "00011",
        "target_video_id": "",
        "study_label": "0",
        "manipulation": "",
        "relative_source_path": (
            "YouTube-real/00011.mp4"
        ),
        "source_label": "1",
    }


def celeb_inference_record():
    return {
        "detector": "xception",
        "checkpoint_sha256": CHECKPOINT,
        "dataset": "Celeb-DF-v2",

        #
        # Deliberately represents the execution /
        # inference role rather than canonical role.
        #
        "role": "test",

        "subgroup": "YouTube-real",
        "study_label": 0,
        "source_label": "1",
        "base_video_id": "00011",
        "relative_source_path": (
            "YouTube-real/00011.mp4"
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
        "video_score": 0.15,
        "inference_status": "ok",
        "failure_stage": "",
        "failure_reason": "",
    }


def celeb_analysis_record():
    return {
        "schema_version": 1,

        "detector": "xception",
        "checkpoint_sha256": CHECKPOINT,

        "dataset": "Celeb-DF-v2",
        "source_split": "test",

        #
        # Canonical role remains authoritative.
        #
        "role": "external_evaluation",

        #
        # Execution role is retained separately.
        #
        "inference_role": "test",

        "video_uid": (
            "Celeb-DF-v2::"
            "YouTube-real/00011.mp4"
        ),
        "pairing_uid": (
            "Celeb-DF-v2::"
            "YouTube-real/00011.mp4"
        ),

        "base_video_id": "00011",
        "source_group_id": "00011",
        "source_video_id": "00011",
        "target_video_id": "",

        "relative_source_path": (
            "YouTube-real/00011.mp4"
        ),

        "study_label": 0,
        "manipulation": "",
        "source_label": "1",

        "bootstrap_stratum": "real",
        "bootstrap_unit_id": "00011",
        "bootstrap_cluster_uid": (
            "Celeb-DF-v2::"
            "external_evaluation::00011"
        ),

        "condition": "CLN",

        "target_frame_budget": 32,
        "clean_valid_frame_count": 3,
        "successful_frame_count": 3,

        "complete_target_frame_set": False,

        "aggregation_method": (
            "unweighted_arithmetic_mean"
        ),

        "video_score": 0.15,

        "inference_status": "ok",
        "analysis_status": "valid_score",
        "analysis_eligible": True,

        "failure_stage": "",
        "failure_reason": "",
    }


def main():
    assert (
        strict_csv_binary_int(
            "0",
            field_name="study_label",
        )
        == 0
    )

    assert (
        strict_csv_binary_int(
            "1",
            field_name="study_label",
        )
        == 1
    )

    invalid_labels = (
        "0.0",
        "1.0",
        "0.7",
        "true",
        "false",
        "",
        0,
        1,
        False,
        True,
        None,
    )

    for value in invalid_labels:
        expect_value_error(
            lambda value=value: (
                strict_csv_binary_int(
                    value,
                    field_name=(
                        "study_label"
                    ),
                )
            ),
            description=(
                "invalid manifest label {!r}"
                .format(
                    value
                )
            ),
        )

    ffpp_manifest = (
        ffpp_manifest_record()
    )

    validate_manifest_record(
        ffpp_manifest
    )

    ffpp_index = (
        build_manifest_index(
            [
                ffpp_manifest
            ]
        )
    )

    validate_manifest_analysis_chain(
        manifest_index=ffpp_index,
        inference_records=[
            ffpp_inference_record()
        ],
        analysis_records=[
            ffpp_analysis_record()
        ],
    )

    #
    # Confirm canonical-role / inference-role separation.
    #
    celeb_manifest = (
        celeb_manifest_record()
    )

    validate_manifest_record(
        celeb_manifest
    )

    celeb_index = (
        build_manifest_index(
            [
                celeb_manifest
            ]
        )
    )

    validate_manifest_analysis_chain(
        manifest_index=celeb_index,
        inference_records=[
            celeb_inference_record()
        ],
        analysis_records=[
            celeb_analysis_record()
        ],
    )

    corrupted_manifest_label = deepcopy(
        ffpp_manifest
    )

    corrupted_manifest_label[
        "study_label"
    ] = "0.7"

    expect_value_error(
        lambda: validate_manifest_record(
            corrupted_manifest_label
        ),
        description=(
            "fractional canonical label"
        ),
    )

    wrong_role = deepcopy(
        ffpp_analysis_record()
    )

    wrong_role[
        "role"
    ] = "validation"

    expect_value_error(
        lambda: validate_manifest_analysis_chain(
            manifest_index=ffpp_index,
            inference_records=[
                ffpp_inference_record()
            ],
            analysis_records=[
                wrong_role
            ],
        ),
        description=(
            "changed canonical role"
        ),
    )

    wrong_source_split = deepcopy(
        ffpp_analysis_record()
    )

    wrong_source_split[
        "source_split"
    ] = "val"

    expect_value_error(
        lambda: validate_manifest_analysis_chain(
            manifest_index=ffpp_index,
            inference_records=[
                ffpp_inference_record()
            ],
            analysis_records=[
                wrong_source_split
            ],
        ),
        description=(
            "changed canonical source_split"
        ),
    )

    wrong_source_video = deepcopy(
        ffpp_analysis_record()
    )

    wrong_source_video[
        "source_video_id"
    ] = "999"

    expect_value_error(
        lambda: validate_manifest_analysis_chain(
            manifest_index=ffpp_index,
            inference_records=[
                ffpp_inference_record()
            ],
            analysis_records=[
                wrong_source_video
            ],
        ),
        description=(
            "changed source_video_id"
        ),
    )

    wrong_pairing_uid = deepcopy(
        ffpp_analysis_record()
    )

    wrong_pairing_uid[
        "pairing_uid"
    ] = "corrupted"

    expect_value_error(
        lambda: validate_manifest_analysis_chain(
            manifest_index=ffpp_index,
            inference_records=[
                ffpp_inference_record()
            ],
            analysis_records=[
                wrong_pairing_uid
            ],
        ),
        description=(
            "changed pairing_uid"
        ),
    )

    wrong_cluster = deepcopy(
        ffpp_analysis_record()
    )

    wrong_cluster[
        "bootstrap_cluster_uid"
    ] = "corrupted"

    expect_value_error(
        lambda: validate_manifest_analysis_chain(
            manifest_index=ffpp_index,
            inference_records=[
                ffpp_inference_record()
            ],
            analysis_records=[
                wrong_cluster
            ],
        ),
        description=(
            "changed bootstrap cluster"
        ),
    )

    wrong_inference_label = deepcopy(
        ffpp_inference_record()
    )

    wrong_inference_label[
        "study_label"
    ] = 1

    expect_value_error(
        lambda: validate_manifest_analysis_chain(
            manifest_index=ffpp_index,
            inference_records=[
                wrong_inference_label
            ],
            analysis_records=[
                ffpp_analysis_record()
            ],
        ),
        description=(
            "inference label differs "
            "from canonical manifest"
        ),
    )

    print(
        "MANIFEST CHAIN CONTRACT VALIDATION"
    )
    print(
        "  strict CSV labels 0/1:          PASSED"
    )
    print(
        "  malformed CSV labels rejected:  PASSED"
    )
    print(
        "  FF++ canonical identity:         PASSED"
    )
    print(
        "  canonical/execution role split: PASSED"
    )
    print(
        "  changed canonical role detected:PASSED"
    )
    print(
        "  changed source split detected:  PASSED"
    )
    print(
        "  changed source linkage detected:PASSED"
    )
    print(
        "  changed pairing UID detected:   PASSED"
    )
    print(
        "  changed bootstrap UID detected: PASSED"
    )
    print(
        "  upstream label drift detected:  PASSED"
    )
    print()
    print(
        "MANIFEST CHAIN CONTRACT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()