from __future__ import annotations

import math
from typing import Any, Dict, Tuple


FFPP_DATASET = "FaceForensics++"
CELEB_DATASET = "Celeb-DF-v2"

SUPPORTED_DATASETS = {
    FFPP_DATASET,
    CELEB_DATASET,
}

SUPPORTED_DETECTORS = {
    "xception",
    "ucf",
    "spsl",
}

SUPPORTED_CONDITIONS = {
    "CLN",
    "RSZ",
    "BLR",
    "H40",
    "PLT",
}

FFPP_MANIPULATIONS = {
    "DeepFakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
}

SUPPORTED_ANALYSIS_ROLES = {
    (
        FFPP_DATASET,
        "validation",
    ),
    (
        FFPP_DATASET,
        "test",
    ),
    (
        CELEB_DATASET,
        "external_evaluation",
    ),
}

VALID_INFERENCE_STATUS = "ok"

ANALYSIS_STATUS_VALID = "valid_score"
ANALYSIS_STATUS_INVALID = "invalid_inference"

EXPECTED_AGGREGATION_METHOD = (
    "unweighted_arithmetic_mean"
)


def manifest_key(
    row: Dict[str, Any],
) -> Tuple[str, str]:
    return (
        str(
            row[
                "dataset"
            ]
        ),
        str(
            row[
                "relative_source_path"
            ]
        ),
    )


def video_uid(
    dataset: str,
    relative_source_path: str,
) -> str:
    return "{}::{}".format(
        dataset,
        relative_source_path,
    )


def analysis_key(
    row: Dict[str, Any],
) -> Tuple[str, str, str, str]:
    return (
        str(
            row[
                "detector"
            ]
        ),
        str(
            row[
                "dataset"
            ]
        ),
        str(
            row[
                "relative_source_path"
            ]
        ),
        str(
            row[
                "condition"
            ]
        ),
    )


def _required_string(
    row: Dict[str, Any],
    field: str,
) -> str:
    if field not in row:
        raise ValueError(
            "Missing required field: {}".format(
                field
            )
        )

    value = str(
        row[
            field
        ]
    ).strip()

    if not value:
        raise ValueError(
            "Required field is empty: {}".format(
                field
            )
        )

    return value


def _required_int(
    row: Dict[str, Any],
    field: str,
) -> int:
    if field not in row:
        raise ValueError(
            "Missing required field: {}".format(
                field
            )
        )

    try:
        return int(
            row[
                field
            ]
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Invalid integer field {}: {!r}".format(
                field,
                row.get(
                    field
                ),
            )
        ) from exc


def _optional_string(
    row: Dict[str, Any],
    field: str,
) -> str:
    value = row.get(
        field,
        "",
    )

    if value is None:
        return ""

    return str(
        value
    ).strip()


def derive_bootstrap_stratum(
    *,
    dataset: str,
    study_label: int,
    manipulation: str,
) -> str:
    if study_label not in (
        0,
        1,
    ):
        raise ValueError(
            "Unsupported study label: {}".format(
                study_label
            )
        )

    if study_label == 0:
        return "real"

    if dataset == FFPP_DATASET:
        if manipulation not in FFPP_MANIPULATIONS:
            raise ValueError(
                "FF++ manipulated record has invalid "
                "manipulation: {!r}".format(
                    manipulation
                )
            )

        return manipulation

    if dataset == CELEB_DATASET:
        return "manipulated"

    raise ValueError(
        "Unsupported dataset: {}".format(
            dataset
        )
    )


def derive_bootstrap_unit_id(
    *,
    dataset: str,
    source_video_id: str,
    base_video_id: str,
) -> str:
    if dataset == FFPP_DATASET:
        if not source_video_id:
            raise ValueError(
                "FF++ record requires source_video_id "
                "for bootstrap clustering."
            )

        return source_video_id

    if dataset == CELEB_DATASET:
        if not base_video_id:
            raise ValueError(
                "Celeb-DF-v2 record requires "
                "base_video_id for bootstrap clustering."
            )

        return base_video_id

    raise ValueError(
        "Unsupported dataset: {}".format(
            dataset
        )
    )


def build_bootstrap_cluster_uid(
    *,
    dataset: str,
    role: str,
    bootstrap_unit_id: str,
) -> str:
    return "{}::{}::{}".format(
        dataset,
        role,
        bootstrap_unit_id,
    )


def _parse_score(
    inference_row: Dict[str, Any],
) -> Any:
    inference_status = _required_string(
        inference_row,
        "inference_status",
    )

    value = inference_row.get(
        "video_score"
    )

    if inference_status != VALID_INFERENCE_STATUS:
        if value is not None:
            raise ValueError(
                "Invalid inference record must not "
                "contain a video score."
            )

        return None

    if value is None:
        raise ValueError(
            "Valid inference record has no "
            "video score."
        )

    try:
        score = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Invalid video score: {!r}".format(
                value
            )
        ) from exc

    if not math.isfinite(
        score
    ):
        raise ValueError(
            "Video score is not finite: {!r}".format(
                score
            )
        )

    return score


def build_analysis_record(
    *,
    inference_row: Dict[str, Any],
    manifest_row: Dict[str, Any],
) -> Dict[str, Any]:
    inference_dataset = _required_string(
        inference_row,
        "dataset",
    )

    inference_path = _required_string(
        inference_row,
        "relative_source_path",
    )

    manifest_dataset = _required_string(
        manifest_row,
        "dataset",
    )

    manifest_path = _required_string(
        manifest_row,
        "relative_source_path",
    )

    if (
        inference_dataset
        != manifest_dataset
        or inference_path
        != manifest_path
    ):
        raise ValueError(
            "Inference/manifest identity mismatch: "
            "{} | {} != {} | {}".format(
                inference_dataset,
                inference_path,
                manifest_dataset,
                manifest_path,
            )
        )

    if (
        manifest_dataset
        not in SUPPORTED_DATASETS
    ):
        raise ValueError(
            "Unsupported dataset: {}".format(
                manifest_dataset
            )
        )

    canonical_role = _required_string(
        manifest_row,
        "role",
    )

    if (
        (
            manifest_dataset,
            canonical_role,
        )
        not in SUPPORTED_ANALYSIS_ROLES
    ):
        raise ValueError(
            "Record is outside the post-training "
            "analysis roles: {} | {}".format(
                manifest_dataset,
                canonical_role,
            )
        )

    detector = _required_string(
        inference_row,
        "detector",
    )

    if detector not in SUPPORTED_DETECTORS:
        raise ValueError(
            "Unsupported detector: {}".format(
                detector
            )
        )

    condition = _required_string(
        inference_row,
        "condition",
    )

    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError(
            "Unsupported condition: {}".format(
                condition
            )
        )

    checkpoint_sha256 = _required_string(
        inference_row,
        "checkpoint_sha256",
    )

    manifest_label = _required_int(
        manifest_row,
        "study_label",
    )

    inference_label = _required_int(
        inference_row,
        "study_label",
    )

    if inference_label != manifest_label:
        raise ValueError(
            "Study-label mismatch for {}: "
            "inference={} manifest={}.".format(
                inference_path,
                inference_label,
                manifest_label,
            )
        )

    manifest_base_video_id = _required_string(
        manifest_row,
        "base_video_id",
    )

    inference_base_video_id = (
        _required_string(
            inference_row,
            "base_video_id",
        )
    )

    if (
        inference_base_video_id
        != manifest_base_video_id
    ):
        raise ValueError(
            "base_video_id mismatch for {}: "
            "inference={} manifest={}.".format(
                inference_path,
                inference_base_video_id,
                manifest_base_video_id,
            )
        )

    source_video_id = _required_string(
        manifest_row,
        "source_video_id",
    )

    source_group_id = _required_string(
        manifest_row,
        "source_group_id",
    )

    target_video_id = _optional_string(
        manifest_row,
        "target_video_id",
    )

    manipulation = _optional_string(
        manifest_row,
        "manipulation",
    )

    source_label = _optional_string(
        manifest_row,
        "source_label",
    )

    source_split = _required_string(
        manifest_row,
        "source_split",
    )

    inference_role = _required_string(
        inference_row,
        "role",
    )

    inference_status = _required_string(
        inference_row,
        "inference_status",
    )

    video_score = _parse_score(
        inference_row
    )

    target_frame_budget = _required_int(
        inference_row,
        "target_frame_budget",
    )

    clean_valid_frame_count = _required_int(
        inference_row,
        "clean_valid_frame_count",
    )

    successful_frame_count = _required_int(
        inference_row,
        "successful_frame_count",
    )

    if target_frame_budget <= 0:
        raise ValueError(
            "target_frame_budget must be positive."
        )

    if (
        clean_valid_frame_count < 0
        or clean_valid_frame_count
        > target_frame_budget
    ):
        raise ValueError(
            "Invalid clean-valid frame count."
        )

    if (
        successful_frame_count < 0
        or successful_frame_count
        > clean_valid_frame_count
    ):
        raise ValueError(
            "Invalid successful frame count."
        )

    if (
        inference_status
        == VALID_INFERENCE_STATUS
    ):
        aggregation_method = (
            _required_string(
                inference_row,
                "aggregation_method",
            )
        )

        if (
            aggregation_method
            != EXPECTED_AGGREGATION_METHOD
        ):
            raise ValueError(
                "Unexpected aggregation method: {}"
                .format(
                    aggregation_method
                )
            )

        if (
            successful_frame_count
            != clean_valid_frame_count
        ):
            raise ValueError(
                "Valid inference record does not "
                "contain all clean-valid positions."
            )

        analysis_status = (
            ANALYSIS_STATUS_VALID
        )

        analysis_eligible = True

    else:
        aggregation_method = (
            inference_row.get(
                "aggregation_method"
            )
        )

        analysis_status = (
            ANALYSIS_STATUS_INVALID
        )

        analysis_eligible = False

    bootstrap_stratum = (
        derive_bootstrap_stratum(
            dataset=manifest_dataset,
            study_label=manifest_label,
            manipulation=manipulation,
        )
    )

    bootstrap_unit_id = (
        derive_bootstrap_unit_id(
            dataset=manifest_dataset,
            source_video_id=(
                source_video_id
            ),
            base_video_id=(
                manifest_base_video_id
            ),
        )
    )

    current_video_uid = video_uid(
        manifest_dataset,
        manifest_path,
    )

    bootstrap_cluster_uid = (
        build_bootstrap_cluster_uid(
            dataset=manifest_dataset,
            role=canonical_role,
            bootstrap_unit_id=(
                bootstrap_unit_id
            ),
        )
    )

    return {
        "schema_version": 1,

        "detector": detector,
        "checkpoint_sha256": (
            checkpoint_sha256
        ),

        "dataset": (
            manifest_dataset
        ),
        "source_split": (
            source_split
        ),
        "role": (
            canonical_role
        ),
        "inference_role": (
            inference_role
        ),

        "video_uid": (
            current_video_uid
        ),
        "pairing_uid": (
            current_video_uid
        ),

        "base_video_id": (
            manifest_base_video_id
        ),
        "source_group_id": (
            source_group_id
        ),
        "source_video_id": (
            source_video_id
        ),
        "target_video_id": (
            target_video_id
        ),

        "relative_source_path": (
            manifest_path
        ),

        "study_label": (
            manifest_label
        ),
        "manipulation": (
            manipulation
        ),
        "source_label": (
            source_label
        ),

        "bootstrap_stratum": (
            bootstrap_stratum
        ),
        "bootstrap_unit_id": (
            bootstrap_unit_id
        ),
        "bootstrap_cluster_uid": (
            bootstrap_cluster_uid
        ),

        "condition": (
            condition
        ),

        "target_frame_budget": (
            target_frame_budget
        ),
        "clean_valid_frame_count": (
            clean_valid_frame_count
        ),
        "successful_frame_count": (
            successful_frame_count
        ),

        "complete_target_frame_set": (
            clean_valid_frame_count
            == target_frame_budget
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
        "analysis_status": (
            analysis_status
        ),
        "analysis_eligible": (
            analysis_eligible
        ),

        "failure_stage": (
            _optional_string(
                inference_row,
                "failure_stage",
            )
        ),
        "failure_reason": (
            _optional_string(
                inference_row,
                "failure_reason",
            )
        ),
    }