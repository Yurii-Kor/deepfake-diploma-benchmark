from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from study.analysis.analysis_schema import (
    CELEB_DATASET,
    FFPP_DATASET,
    FFPP_MANIPULATIONS,
    SUPPORTED_CONDITIONS,
)


FFPP_MANIPULATION_ORDER = (
    "DeepFakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
)

FFPP_POOLED_GROUP = "pooled"
CELEB_POOLED_GROUP = "pooled"

GROUP_TYPE_MANIPULATION_SPECIFIC = (
    "manipulation_specific"
)

GROUP_TYPE_POOLED = "pooled"

CONDITION_ORDER = {
    "CLN": 0,
    "RSZ": 1,
    "BLR": 2,
    "H40": 3,
    "PLT": 4,
}

GROUP_ORDER = {
    "DeepFakes": 0,
    "Face2Face": 1,
    "FaceSwap": 2,
    "NeuralTextures": 3,
    "pooled": 4,
}


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


def _binary_label(
    row: Dict[str, Any],
) -> int:
    try:
        label = int(
            row[
                "study_label"
            ]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Invalid or missing study_label."
        ) from exc

    if label not in (
        0,
        1,
    ):
        raise ValueError(
            "study_label must be 0 or 1, got {}."
            .format(
                label
            )
        )

    return label


def decision_record_key(
    row: Dict[str, Any],
) -> Tuple[str, str, str, str]:
    return (
        _required_string(
            row,
            "detector",
        ),
        _required_string(
            row,
            "dataset",
        ),
        _required_string(
            row,
            "relative_source_path",
        ),
        _required_string(
            row,
            "condition",
        ),
    )


def is_valid_metric_record(
    row: Dict[str, Any],
) -> bool:
    if (
        row.get(
            "analysis_eligible"
        )
        is not True
    ):
        return False

    if (
        row.get(
            "video_score"
        )
        is None
    ):
        return False

    if (
        row.get(
            "decision_status"
        )
        != "decided"
    ):
        return False

    try:
        decision = int(
            row.get(
                "decision"
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return False

    return decision in (
        0,
        1,
    )


def _validate_decision_record(
    row: Dict[str, Any],
):
    detector = _required_string(
        row,
        "detector",
    )

    checkpoint_sha256 = (
        _required_string(
            row,
            "checkpoint_sha256",
        )
    )

    dataset = _required_string(
        row,
        "dataset",
    )

    role = _required_string(
        row,
        "role",
    )

    relative_source_path = (
        _required_string(
            row,
            "relative_source_path",
        )
    )

    condition = _required_string(
        row,
        "condition",
    )

    if (
        condition
        not in SUPPORTED_CONDITIONS
    ):
        raise ValueError(
            "Unsupported condition: {}".format(
                condition
            )
        )

    label = _binary_label(
        row
    )

    manipulation = _optional_string(
        row,
        "manipulation",
    )

    if dataset == FFPP_DATASET:
        if role not in (
            "validation",
            "test",
        ):
            raise ValueError(
                "Unexpected FF++ analysis role: {}"
                .format(
                    role
                )
            )

        if label == 0:
            if manipulation:
                raise ValueError(
                    "FF++ real record must not "
                    "have a manipulation label: {}"
                    .format(
                        relative_source_path
                    )
                )

        else:
            if (
                manipulation
                not in FFPP_MANIPULATIONS
            ):
                raise ValueError(
                    "FF++ manipulated record has "
                    "invalid manipulation {!r}: {}"
                    .format(
                        manipulation,
                        relative_source_path,
                    )
                )

    elif dataset == CELEB_DATASET:
        if role != "external_evaluation":
            raise ValueError(
                "Unexpected Celeb-DF-v2 "
                "analysis role: {}".format(
                    role
                )
            )

    else:
        raise ValueError(
            "Unsupported evaluation dataset: {}"
            .format(
                dataset
            )
        )

    eligible = (
        row.get(
            "analysis_eligible"
        )
        is True
    )

    score = row.get(
        "video_score"
    )

    decision_status = row.get(
        "decision_status"
    )

    decision = row.get(
        "decision"
    )

    if eligible:
        if score is None:
            raise ValueError(
                "Eligible analysis record has "
                "no video score: {}".format(
                    relative_source_path
                )
            )

        if decision_status != "decided":
            raise ValueError(
                "Eligible analysis record has "
                "no frozen-threshold decision: {}"
                .format(
                    relative_source_path
                )
            )

        try:
            normalized_decision = int(
                decision
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "Eligible analysis record has "
                "invalid decision: {}".format(
                    relative_source_path
                )
            ) from exc

        if normalized_decision not in (
            0,
            1,
        ):
            raise ValueError(
                "Decision must be binary: {}"
                .format(
                    relative_source_path
                )
            )

    else:
        if score is not None:
            raise ValueError(
                "Ineligible analysis record must "
                "not contain a video score: {}"
                .format(
                    relative_source_path
                )
            )

        if decision is not None:
            raise ValueError(
                "Ineligible analysis record must "
                "not contain a decision: {}"
                .format(
                    relative_source_path
                )
            )

        if decision_status != "not_scored":
            raise ValueError(
                "Ineligible analysis record must "
                "have decision_status=not_scored: {}"
                .format(
                    relative_source_path
                )
            )

    #
    # Variables are intentionally read above even where
    # the local function does not otherwise use them.
    # Their presence is part of the decision-record
    # contract.
    #
    _ = (
        detector,
        checkpoint_sha256,
    )


def _build_metric_group(
    *,
    detector: str,
    checkpoint_sha256: str,
    dataset: str,
    role: str,
    condition: str,
    group_name: str,
    group_type: str,
    real_records: List[
        Dict[str, Any]
    ],
    manipulated_records: List[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    valid_real_records = [
        row
        for row in real_records
        if is_valid_metric_record(
            row
        )
    ]

    valid_manipulated_records = [
        row
        for row in manipulated_records
        if is_valid_metric_record(
            row
        )
    ]

    return {
        "detector": (
            detector
        ),

        "checkpoint_sha256": (
            checkpoint_sha256
        ),

        "dataset": (
            dataset
        ),

        "role": (
            role
        ),

        "condition": (
            condition
        ),

        "group_name": (
            group_name
        ),

        "group_type": (
            group_type
        ),

        "real_records": (
            real_records
        ),

        "manipulated_records": (
            manipulated_records
        ),

        "valid_real_records": (
            valid_real_records
        ),

        "valid_manipulated_records": (
            valid_manipulated_records
        ),

        "n_real_nominal": (
            len(
                real_records
            )
        ),

        "n_manipulated_nominal": (
            len(
                manipulated_records
            )
        ),

        "n_real_valid": (
            len(
                valid_real_records
            )
        ),

        "n_manipulated_valid": (
            len(
                valid_manipulated_records
            )
        ),

        "n_real_invalid": (
            len(
                real_records
            )
            - len(
                valid_real_records
            )
        ),

        "n_manipulated_invalid": (
            len(
                manipulated_records
            )
            - len(
                valid_manipulated_records
            )
        ),
    }


def metric_group_key(
    group: Dict[str, Any],
) -> Tuple[
    str,
    str,
    str,
    str,
    str,
]:
    return (
        str(
            group[
                "detector"
            ]
        ),
        str(
            group[
                "dataset"
            ]
        ),
        str(
            group[
                "role"
            ]
        ),
        str(
            group[
                "condition"
            ]
        ),
        str(
            group[
                "group_name"
            ]
        ),
    )


def build_primary_metric_groups(
    records: Iterable[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    rows = list(
        records
    )

    if not rows:
        raise ValueError(
            "Decision-record input is empty."
        )

    seen_record_keys = set()
    checkpoint_by_detector = {}

    contexts = defaultdict(
        list
    )

    for row in rows:
        _validate_decision_record(
            row
        )

        dataset = str(
            row[
                "dataset"
            ]
        )

        role = str(
            row[
                "role"
            ]
        )

        #
        # FF++ validation belongs to the threshold and
        # diagnostic-threshold branches, not to the
        # primary held-out metric groups.
        #
        if (
            dataset
            == FFPP_DATASET
            and role
            == "validation"
        ):
            continue

        record_key = (
            decision_record_key(
                row
            )
        )

        if (
            record_key
            in seen_record_keys
        ):
            raise ValueError(
                "Duplicate primary evaluation "
                "decision record: {}".format(
                    record_key
                )
            )

        seen_record_keys.add(
            record_key
        )

        detector = str(
            row[
                "detector"
            ]
        )

        checkpoint_sha256 = str(
            row[
                "checkpoint_sha256"
            ]
        )

        previous_checkpoint = (
            checkpoint_by_detector.get(
                detector
            )
        )

        if (
            previous_checkpoint
            is not None
            and previous_checkpoint
            != checkpoint_sha256
        ):
            raise ValueError(
                "Detector {} has multiple "
                "checkpoints in metric input."
                .format(
                    detector
                )
            )

        checkpoint_by_detector[
            detector
        ] = checkpoint_sha256

        context_key = (
            detector,
            checkpoint_sha256,
            dataset,
            role,
            str(
                row[
                    "condition"
                ]
            ),
        )

        contexts[
            context_key
        ].append(
            row
        )

    groups = []

    for (
        detector,
        checkpoint_sha256,
        dataset,
        role,
        condition,
    ), context_rows in (
        contexts.items()
    ):
        real_records = [
            row
            for row in context_rows
            if (
                _binary_label(
                    row
                )
                == 0
            )
        ]

        manipulated_records = [
            row
            for row in context_rows
            if (
                _binary_label(
                    row
                )
                == 1
            )
        ]

        if dataset == FFPP_DATASET:
            if role != "test":
                raise ValueError(
                    "Primary FF++ metric group "
                    "must use role=test."
                )

            for manipulation in (
                FFPP_MANIPULATION_ORDER
            ):
                method_records = [
                    row
                    for row in (
                        manipulated_records
                    )
                    if (
                        str(
                            row[
                                "manipulation"
                            ]
                        )
                        == manipulation
                    )
                ]

                groups.append(
                    _build_metric_group(
                        detector=detector,
                        checkpoint_sha256=(
                            checkpoint_sha256
                        ),
                        dataset=dataset,
                        role=role,
                        condition=condition,
                        group_name=(
                            manipulation
                        ),
                        group_type=(
                            GROUP_TYPE_MANIPULATION_SPECIFIC
                        ),
                        real_records=(
                            list(
                                real_records
                            )
                        ),
                        manipulated_records=(
                            method_records
                        ),
                    )
                )

            groups.append(
                _build_metric_group(
                    detector=detector,
                    checkpoint_sha256=(
                        checkpoint_sha256
                    ),
                    dataset=dataset,
                    role=role,
                    condition=condition,
                    group_name=(
                        FFPP_POOLED_GROUP
                    ),
                    group_type=(
                        GROUP_TYPE_POOLED
                    ),
                    real_records=(
                        list(
                            real_records
                        )
                    ),
                    manipulated_records=(
                        list(
                            manipulated_records
                        )
                    ),
                )
            )

        elif dataset == CELEB_DATASET:
            if (
                role
                != "external_evaluation"
            ):
                raise ValueError(
                    "Primary Celeb-DF-v2 metric "
                    "group must use "
                    "role=external_evaluation."
                )

            groups.append(
                _build_metric_group(
                    detector=detector,
                    checkpoint_sha256=(
                        checkpoint_sha256
                    ),
                    dataset=dataset,
                    role=role,
                    condition=condition,
                    group_name=(
                        CELEB_POOLED_GROUP
                    ),
                    group_type=(
                        GROUP_TYPE_POOLED
                    ),
                    real_records=(
                        list(
                            real_records
                        )
                    ),
                    manipulated_records=(
                        list(
                            manipulated_records
                        )
                    ),
                )
            )

        else:
            raise ValueError(
                "Unsupported dataset in metric "
                "group construction: {}".format(
                    dataset
                )
            )

    seen_group_keys = set()

    for group in groups:
        key = metric_group_key(
            group
        )

        if key in seen_group_keys:
            raise ValueError(
                "Duplicate metric group: {}"
                .format(
                    key
                )
            )

        seen_group_keys.add(
            key
        )

    groups.sort(
        key=lambda group: (
            group[
                "detector"
            ],
            group[
                "dataset"
            ],
            group[
                "role"
            ],
            CONDITION_ORDER[
                group[
                    "condition"
                ]
            ],
            GROUP_ORDER[
                group[
                    "group_name"
                ]
            ],
        )
    )

    return groups