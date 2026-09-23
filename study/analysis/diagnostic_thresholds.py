from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

from study.analysis.analysis_schema import (
    FFPP_DATASET,
)
from study.analysis.operating_point import (
    PRIMARY_TARGET_FPR,
    select_primary_low_fpr_threshold,
)


DEGRADED_CONDITIONS = (
    "RSZ",
    "BLR",
    "H40",
    "PLT",
)

CONDITION_ORDER = {
    "RSZ": 0,
    "BLR": 1,
    "H40": 2,
    "PLT": 3,
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


def _study_label(
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
            "study_label must be 0 or 1."
        )

    return label


def _primary_threshold_index(
    artifact: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    if (
        artifact.get(
            "artifact_type"
        )
        != "primary_operating_thresholds"
    ):
        raise ValueError(
            "Unexpected primary threshold "
            "artifact type."
        )

    try:
        artifact_target_fpr = float(
            artifact[
                "target_fpr"
            ]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Primary threshold artifact has "
            "invalid target_fpr."
        ) from exc

    if not math.isclose(
        artifact_target_fpr,
        PRIMARY_TARGET_FPR,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "Primary threshold artifact does not "
            "use the frozen study target FPR."
        )

    thresholds = artifact.get(
        "thresholds"
    )

    if not isinstance(
        thresholds,
        list,
    ):
        raise ValueError(
            "Primary threshold artifact has no "
            "threshold list."
        )

    index = {}

    for row in thresholds:
        detector = _required_string(
            row,
            "detector",
        )

        if detector in index:
            raise ValueError(
                "Duplicate primary threshold for "
                "detector {}.".format(
                    detector
                )
            )

        index[
            detector
        ] = row

    if not index:
        raise ValueError(
            "Primary threshold artifact is empty."
        )

    return index


def _diagnostic_validation_rows(
    records: Iterable[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    selected = []

    for row in records:
        if (
            row.get(
                "dataset"
            )
            != FFPP_DATASET
        ):
            continue

        if (
            row.get(
                "role"
            )
            != "validation"
        ):
            continue

        if (
            _study_label(
                row
            )
            != 0
        ):
            continue

        condition = str(
            row.get(
                "condition",
                "",
            )
        )

        if (
            condition
            not in DEGRADED_CONDITIONS
        ):
            continue

        selected.append(
            row
        )

    return selected


def compute_diagnostic_thresholds(
    *,
    analysis_records: Iterable[
        Dict[str, Any]
    ],
    primary_threshold_artifact: Dict[
        str,
        Any,
    ],
) -> List[Dict[str, Any]]:
    rows = _diagnostic_validation_rows(
        analysis_records
    )

    if not rows:
        raise ValueError(
            "No processed FF++ validation real "
            "records are available for diagnostic "
            "threshold estimation."
        )

    primary_index = (
        _primary_threshold_index(
            primary_threshold_artifact
        )
    )

    contexts = {}

    for row in rows:
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

        condition = _required_string(
            row,
            "condition",
        )

        if (
            condition
            not in DEGRADED_CONDITIONS
        ):
            raise ValueError(
                "Unexpected diagnostic condition: {}"
                .format(
                    condition
                )
            )

        context_key = (
            detector,
            checkpoint_sha256,
            condition,
        )

        contexts.setdefault(
            context_key,
            [],
        ).append(
            row
        )

    output = []

    for (
        detector,
        checkpoint_sha256,
        condition,
    ), condition_rows in (
        contexts.items()
    ):
        primary = primary_index.get(
            detector
        )

        if primary is None:
            raise ValueError(
                "No primary clean threshold exists "
                "for detector {}.".format(
                    detector
                )
            )

        primary_checkpoint = (
            _required_string(
                primary,
                "checkpoint_sha256",
            )
        )

        if (
            primary_checkpoint
            != checkpoint_sha256
        ):
            raise ValueError(
                "Diagnostic records and primary "
                "threshold use different checkpoints "
                "for detector {}.".format(
                    detector
                )
            )

        if (
            primary.get(
                "calibration_dataset"
            )
            != FFPP_DATASET
            or primary.get(
                "calibration_role"
            )
            != "validation"
            or primary.get(
                "calibration_condition"
            )
            != "CLN"
            or primary.get(
                "calibration_class"
            )
            != "real_only"
        ):
            raise ValueError(
                "Primary threshold provenance is "
                "incompatible with the study "
                "diagnostic-threshold protocol."
            )

        pairing_uids = [
            _required_string(
                row,
                "pairing_uid",
            )
            for row in condition_rows
        ]

        if (
            len(
                pairing_uids
            )
            != len(
                set(
                    pairing_uids
                )
            )
        ):
            raise ValueError(
                "Duplicate validation real video "
                "for detector {} condition {}."
                .format(
                    detector,
                    condition,
                )
            )

        valid_rows = [
            row
            for row in condition_rows
            if (
                row.get(
                    "analysis_eligible"
                )
                is True
                and row.get(
                    "video_score"
                )
                is not None
            )
        ]

        if not valid_rows:
            raise ValueError(
                "Detector {} condition {} has no "
                "valid validation real scores."
                .format(
                    detector,
                    condition,
                )
            )

        diagnostic = (
            select_primary_low_fpr_threshold(
                real_scores=[
                    row[
                        "video_score"
                    ]
                    for row in valid_rows
                ],
                target_fpr=(
                    PRIMARY_TARGET_FPR
                ),
            )
        )

        primary_threshold = float(
            primary[
                "threshold_value"
            ]
        )

        diagnostic_threshold = float(
            diagnostic[
                "threshold_value"
            ]
        )

        displacement = (
            diagnostic_threshold
            - primary_threshold
        )

        diagnostic_id = (
            "{}_ffpp_validation_{}_fpr005_{}"
            .format(
                detector,
                condition.lower(),
                checkpoint_sha256[
                    :12
                ],
            )
        )

        result = {
            "schema_version": 1,

            "diagnostic_threshold_id": (
                diagnostic_id
            ),

            "detector": (
                detector
            ),

            "checkpoint_sha256": (
                checkpoint_sha256
            ),

            "dataset": (
                FFPP_DATASET
            ),

            "role": (
                "validation"
            ),

            "condition": (
                condition
            ),

            "calibration_class": (
                "real_only"
            ),

            "target_fpr": (
                PRIMARY_TARGET_FPR
            ),

            "nominal_real_records": (
                len(
                    condition_rows
                )
            ),

            "valid_real_records": (
                len(
                    valid_rows
                )
            ),

            "excluded_invalid_real_records": (
                len(
                    condition_rows
                )
                - len(
                    valid_rows
                )
            ),

            "primary_threshold_id": (
                primary[
                    "threshold_id"
                ]
            ),

            "primary_clean_threshold": (
                primary_threshold
            ),

            "diagnostic_threshold": (
                diagnostic_threshold
            ),

            "threshold_displacement": (
                displacement
            ),

            "displacement_definition": (
                "diagnostic_minus_primary_clean"
            ),

            "clean_threshold_reestimated": (
                False
            ),

            "processed_failure_changes_primary_threshold": (
                False
            ),
        }

        for field in (
            "realized_fpr",
            "false_positives",
            "max_allowed_false_positives",
            "candidate_count",
            "selection_rule",
            "candidate_rule",
            "tie_rule",
            "decision_rule",
            "score_direction",
        ):
            result[
                "diagnostic_{}".format(
                    field
                )
            ] = diagnostic[
                field
            ]

        output.append(
            result
        )

    output.sort(
        key=lambda row: (
            row[
                "detector"
            ],
            CONDITION_ORDER[
                row[
                    "condition"
                ]
            ],
        )
    )

    seen = set()

    for row in output:
        key = (
            row[
                "detector"
            ],
            row[
                "condition"
            ],
        )

        if key in seen:
            raise ValueError(
                "Duplicate diagnostic threshold "
                "result: {}".format(
                    key
                )
            )

        seen.add(
            key
        )

    return output