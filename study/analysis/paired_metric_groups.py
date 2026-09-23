from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from study.analysis.metric_groups import (
    build_primary_metric_groups,
)


CLEAN_CONDITION = "CLN"

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


def _pairing_uid(
    row: Dict[str, Any],
) -> str:
    value = str(
        row.get(
            "pairing_uid",
            "",
        )
    ).strip()

    if not value:
        raise ValueError(
            "Metric record has no pairing_uid."
        )

    return value


def _index_valid_records(
    records: Iterable[
        Dict[str, Any]
    ],
    *,
    context: str,
) -> Dict[
    str,
    Dict[str, Any],
]:
    index = {}

    for row in records:
        uid = _pairing_uid(
            row
        )

        if uid in index:
            raise ValueError(
                "Duplicate pairing_uid in {}: {}"
                .format(
                    context,
                    uid,
                )
            )

        index[
            uid
        ] = row

    return index


def _assert_pair_compatible(
    *,
    clean_row: Dict[str, Any],
    degraded_row: Dict[str, Any],
):
    fields = (
        "detector",
        "checkpoint_sha256",
        "dataset",
        "role",
        "pairing_uid",
        "relative_source_path",
        "study_label",
        "manipulation",
        "threshold_id",
    )

    for field in fields:
        if (
            clean_row.get(
                field
            )
            != degraded_row.get(
                field
            )
        ):
            raise ValueError(
                "CLN/degraded record mismatch "
                "for field {}: {!r} != {!r}"
                .format(
                    field,
                    clean_row.get(
                        field
                    ),
                    degraded_row.get(
                        field
                    ),
                )
            )

    clean_threshold = float(
        clean_row[
            "threshold_value"
        ]
    )

    degraded_threshold = float(
        degraded_row[
            "threshold_value"
        ]
    )

    if (
        clean_threshold
        != degraded_threshold
    ):
        raise ValueError(
            "Frozen threshold changed between "
            "CLN and degraded condition."
        )


def _paired_class_records(
    *,
    clean_records: List[
        Dict[str, Any]
    ],
    degraded_records: List[
        Dict[str, Any]
    ],
    context: str,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    clean_index = (
        _index_valid_records(
            clean_records,
            context=(
                "{} CLN".format(
                    context
                )
            ),
        )
    )

    degraded_index = (
        _index_valid_records(
            degraded_records,
            context=(
                "{} degraded".format(
                    context
                )
            ),
        )
    )

    shared_uids = sorted(
        set(
            clean_index
        )
        & set(
            degraded_index
        )
    )

    paired_clean = []
    paired_degraded = []

    for uid in shared_uids:
        clean_row = (
            clean_index[
                uid
            ]
        )

        degraded_row = (
            degraded_index[
                uid
            ]
        )

        _assert_pair_compatible(
            clean_row=clean_row,
            degraded_row=degraded_row,
        )

        paired_clean.append(
            clean_row
        )

        paired_degraded.append(
            degraded_row
        )

    return (
        paired_clean,
        paired_degraded,
    )


def _base_group_key(
    group: Dict[str, Any],
) -> Tuple[
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
                "group_name"
            ]
        ),
    )


def paired_metric_group_key(
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
                "group_name"
            ]
        ),
        str(
            group[
                "degraded_condition"
            ]
        ),
    )


def build_paired_metric_groups(
    records: Iterable[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    metric_groups = (
        build_primary_metric_groups(
            records
        )
    )

    by_context = {}

    for group in metric_groups:
        key = (
            _base_group_key(
                group
            )
            + (
                str(
                    group[
                        "condition"
                    ]
                ),
            )
        )

        if key in by_context:
            raise ValueError(
                "Duplicate metric-group context: {}"
                .format(
                    key
                )
            )

        by_context[
            key
        ] = group

    base_keys = sorted(
        {
            _base_group_key(
                group
            )
            for group in (
                metric_groups
            )
        }
    )

    output = []

    for base_key in base_keys:
        clean_key = (
            base_key
            + (
                CLEAN_CONDITION,
            )
        )

        clean_group = (
            by_context.get(
                clean_key
            )
        )

        #
        # A context that has only a degraded condition
        # is invalid for clean-to-degraded analysis.
        #
        if clean_group is None:
            degraded_exists = any(
                (
                    base_key
                    + (
                        condition,
                    )
                )
                in by_context
                for condition in (
                    DEGRADED_CONDITIONS
                )
            )

            if degraded_exists:
                raise ValueError(
                    "Degraded metric group exists "
                    "without matching CLN group: {}"
                    .format(
                        base_key
                    )
                )

            continue

        for degraded_condition in (
            DEGRADED_CONDITIONS
        ):
            degraded_key = (
                base_key
                + (
                    degraded_condition,
                )
            )

            degraded_group = (
                by_context.get(
                    degraded_key
                )
            )

            #
            # Not every smoke necessarily contains all
            # four degraded conditions.
            #
            if degraded_group is None:
                continue

            if (
                clean_group[
                    "checkpoint_sha256"
                ]
                != degraded_group[
                    "checkpoint_sha256"
                ]
            ):
                raise ValueError(
                    "Checkpoint changed between CLN "
                    "and {} for {}."
                    .format(
                        degraded_condition,
                        base_key,
                    )
                )

            (
                clean_real,
                degraded_real,
            ) = _paired_class_records(
                clean_records=(
                    clean_group[
                        "valid_real_records"
                    ]
                ),
                degraded_records=(
                    degraded_group[
                        "valid_real_records"
                    ]
                ),
                context=(
                    "{} real {}"
                    .format(
                        base_key,
                        degraded_condition,
                    )
                ),
            )

            (
                clean_manipulated,
                degraded_manipulated,
            ) = _paired_class_records(
                clean_records=(
                    clean_group[
                        "valid_manipulated_records"
                    ]
                ),
                degraded_records=(
                    degraded_group[
                        "valid_manipulated_records"
                    ]
                ),
                context=(
                    "{} manipulated {}"
                    .format(
                        base_key,
                        degraded_condition,
                    )
                ),
            )

            output.append(
                {
                    "detector": (
                        clean_group[
                            "detector"
                        ]
                    ),

                    "checkpoint_sha256": (
                        clean_group[
                            "checkpoint_sha256"
                        ]
                    ),

                    "dataset": (
                        clean_group[
                            "dataset"
                        ]
                    ),

                    "role": (
                        clean_group[
                            "role"
                        ]
                    ),

                    "group_name": (
                        clean_group[
                            "group_name"
                        ]
                    ),

                    "group_type": (
                        clean_group[
                            "group_type"
                        ]
                    ),

                    "clean_condition": (
                        CLEAN_CONDITION
                    ),

                    "degraded_condition": (
                        degraded_condition
                    ),

                    "clean_real_records": (
                        clean_real
                    ),

                    "degraded_real_records": (
                        degraded_real
                    ),

                    "clean_manipulated_records": (
                        clean_manipulated
                    ),

                    "degraded_manipulated_records": (
                        degraded_manipulated
                    ),

                    "n_real_clean_valid": (
                        clean_group[
                            "n_real_valid"
                        ]
                    ),

                    "n_real_degraded_valid": (
                        degraded_group[
                            "n_real_valid"
                        ]
                    ),

                    "n_real_paired": (
                        len(
                            clean_real
                        )
                    ),

                    "n_manipulated_clean_valid": (
                        clean_group[
                            "n_manipulated_valid"
                        ]
                    ),

                    "n_manipulated_degraded_valid": (
                        degraded_group[
                            "n_manipulated_valid"
                        ]
                    ),

                    "n_manipulated_paired": (
                        len(
                            clean_manipulated
                        )
                    ),

                    "n_real_excluded_from_pair": (
                        (
                            clean_group[
                                "n_real_valid"
                            ]
                            - len(
                                clean_real
                            )
                        )
                    ),

                    "n_manipulated_excluded_from_pair": (
                        (
                            clean_group[
                                "n_manipulated_valid"
                            ]
                            - len(
                                clean_manipulated
                            )
                        )
                    ),
                }
            )

    seen = set()

    for group in output:
        key = paired_metric_group_key(
            group
        )

        if key in seen:
            raise ValueError(
                "Duplicate paired metric group: {}"
                .format(
                    key
                )
            )

        seen.add(
            key
        )

    output.sort(
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
            group[
                "group_name"
            ],
            CONDITION_ORDER[
                group[
                    "degraded_condition"
                ]
            ],
        )
    )

    return output