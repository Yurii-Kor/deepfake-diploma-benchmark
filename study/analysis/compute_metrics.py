from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from study.analysis.metric_groups import (
    build_primary_metric_groups,
)
from study.analysis.metrics import (
    binary_auc,
    fixed_threshold_metrics,
)
from study.analysis.paired_metric_groups import (
    build_paired_metric_groups,
)


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
            "study_label must be 0 or 1."
        )

    return label


def _decision(
    row: Dict[str, Any],
) -> int:
    try:
        decision = int(
            row[
                "decision"
            ]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Invalid or missing decision."
        ) from exc

    if decision not in (
        0,
        1,
    ):
        raise ValueError(
            "Decision must be 0 or 1."
        )

    return decision


def _score(
    row: Dict[str, Any],
) -> float:
    try:
        score = float(
            row[
                "video_score"
            ]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Invalid or missing video_score."
        ) from exc

    return score


def _threshold_identity(
    records: Iterable[
        Dict[str, Any]
    ],
) -> Tuple[str, float]:
    rows = list(
        records
    )

    if not rows:
        raise ValueError(
            "Threshold identity requires "
            "at least one record."
        )

    threshold_ids = {
        str(
            row[
                "threshold_id"
            ]
        )
        for row in rows
    }

    if len(
        threshold_ids
    ) != 1:
        raise ValueError(
            "Metric population contains multiple "
            "threshold IDs."
        )

    threshold_values = {
        float(
            row[
                "threshold_value"
            ]
        )
        for row in rows
    }

    if len(
        threshold_values
    ) != 1:
        raise ValueError(
            "Metric population contains multiple "
            "threshold values."
        )

    threshold_id = next(
        iter(
            threshold_ids
        )
    )

    threshold_value = next(
        iter(
            threshold_values
        )
    )

    return (
        threshold_id,
        threshold_value,
    )


def _validate_decisions_against_threshold(
    *,
    records: Iterable[
        Dict[str, Any]
    ],
    threshold_value: float,
):
    for row in records:
        score = _score(
            row
        )

        decision = _decision(
            row
        )

        expected = (
            1
            if score >= threshold_value
            else 0
        )

        if decision != expected:
            raise ValueError(
                "Stored decision is inconsistent "
                "with frozen threshold for {}."
                .format(
                    row.get(
                        "relative_source_path",
                        "<unknown>",
                    )
                )
            )


def compute_metric_values(
    *,
    real_records: List[
        Dict[str, Any]
    ],
    manipulated_records: List[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    if not real_records:
        raise ValueError(
            "Metric computation requires at "
            "least one valid real video."
        )

    if not manipulated_records:
        raise ValueError(
            "Metric computation requires at "
            "least one valid manipulated video."
        )

    for row in real_records:
        if _binary_label(
            row
        ) != 0:
            raise ValueError(
                "Real metric population contains "
                "a manipulated label."
            )

    for row in manipulated_records:
        if _binary_label(
            row
        ) != 1:
            raise ValueError(
                "Manipulated metric population "
                "contains a real label."
            )

    all_records = (
        list(
            real_records
        )
        + list(
            manipulated_records
        )
    )

    (
        threshold_id,
        threshold_value,
    ) = _threshold_identity(
        all_records
    )

    _validate_decisions_against_threshold(
        records=all_records,
        threshold_value=(
            threshold_value
        ),
    )

    auc = binary_auc(
        real_scores=[
            _score(
                row
            )
            for row in real_records
        ],
        manipulated_scores=[
            _score(
                row
            )
            for row in manipulated_records
        ],
    )

    fixed = fixed_threshold_metrics(
        labels=(
            [
                0
                for _ in real_records
            ]
            + [
                1
                for _ in manipulated_records
            ]
        ),
        decisions=(
            [
                _decision(
                    row
                )
                for row in real_records
            ]
            + [
                _decision(
                    row
                )
                for row in manipulated_records
            ]
        ),
    )

    result = {
        "threshold_id": (
            threshold_id
        ),

        "threshold_value": (
            threshold_value
        ),

        "auc": (
            auc
        ),
    }

    result.update(
        fixed
    )

    return result


def compute_absolute_metrics(
    records: Iterable[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    groups = (
        build_primary_metric_groups(
            records
        )
    )

    output = []

    for group in groups:
        #
        # A partial synthetic fixture may not contain
        # every FF++ manipulation method. Such an absent
        # population is not a metric result.
        #
        if (
            group[
                "n_real_nominal"
            ]
            == 0
            or group[
                "n_manipulated_nominal"
            ]
            == 0
        ):
            continue

        if (
            group[
                "n_real_valid"
            ]
            == 0
        ):
            raise ValueError(
                "Metric group has nominal real "
                "videos but no valid real scores: "
                "{} | {} | {} | {}"
                .format(
                    group[
                        "detector"
                    ],
                    group[
                        "dataset"
                    ],
                    group[
                        "condition"
                    ],
                    group[
                        "group_name"
                    ],
                )
            )

        if (
            group[
                "n_manipulated_valid"
            ]
            == 0
        ):
            raise ValueError(
                "Metric group has nominal manipulated "
                "videos but no valid manipulated scores: "
                "{} | {} | {} | {}"
                .format(
                    group[
                        "detector"
                    ],
                    group[
                        "dataset"
                    ],
                    group[
                        "condition"
                    ],
                    group[
                        "group_name"
                    ],
                )
            )

        values = compute_metric_values(
            real_records=(
                group[
                    "valid_real_records"
                ]
            ),
            manipulated_records=(
                group[
                    "valid_manipulated_records"
                ]
            ),
        )

        result = {
            "schema_version": 1,
            "metric_type": (
                "absolute"
            ),

            "detector": (
                group[
                    "detector"
                ]
            ),

            "checkpoint_sha256": (
                group[
                    "checkpoint_sha256"
                ]
            ),

            "dataset": (
                group[
                    "dataset"
                ]
            ),

            "role": (
                group[
                    "role"
                ]
            ),

            "condition": (
                group[
                    "condition"
                ]
            ),

            "group_name": (
                group[
                    "group_name"
                ]
            ),

            "group_type": (
                group[
                    "group_type"
                ]
            ),

            "n_real_nominal": (
                group[
                    "n_real_nominal"
                ]
            ),

            "n_manipulated_nominal": (
                group[
                    "n_manipulated_nominal"
                ]
            ),

            "n_real_valid": (
                group[
                    "n_real_valid"
                ]
            ),

            "n_manipulated_valid": (
                group[
                    "n_manipulated_valid"
                ]
            ),

            "n_real_invalid": (
                group[
                    "n_real_invalid"
                ]
            ),

            "n_manipulated_invalid": (
                group[
                    "n_manipulated_invalid"
                ]
            ),
        }

        result.update(
            values
        )

        output.append(
            result
        )

    return output


def compute_paired_degradation_metrics(
    records: Iterable[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    groups = (
        build_paired_metric_groups(
            records
        )
    )

    output = []

    for group in groups:
        #
        # Empty manipulation-specific groups can occur
        # only in deliberately partial test fixtures.
        #
        if (
            group[
                "n_manipulated_clean_valid"
            ]
            == 0
            and group[
                "n_manipulated_degraded_valid"
            ]
            == 0
        ):
            continue

        if (
            group[
                "n_real_paired"
            ]
            == 0
        ):
            raise ValueError(
                "No paired real videos remain for "
                "{} | {} | {}."
                .format(
                    group[
                        "dataset"
                    ],
                    group[
                        "group_name"
                    ],
                    group[
                        "degraded_condition"
                    ],
                )
            )

        if (
            group[
                "n_manipulated_paired"
            ]
            == 0
        ):
            raise ValueError(
                "No paired manipulated videos remain "
                "for {} | {} | {}."
                .format(
                    group[
                        "dataset"
                    ],
                    group[
                        "group_name"
                    ],
                    group[
                        "degraded_condition"
                    ],
                )
            )

        clean_values = compute_metric_values(
            real_records=(
                group[
                    "clean_real_records"
                ]
            ),
            manipulated_records=(
                group[
                    "clean_manipulated_records"
                ]
            ),
        )

        degraded_values = (
            compute_metric_values(
                real_records=(
                    group[
                        "degraded_real_records"
                    ]
                ),
                manipulated_records=(
                    group[
                        "degraded_manipulated_records"
                    ]
                ),
            )
        )

        if (
            clean_values[
                "threshold_id"
            ]
            != degraded_values[
                "threshold_id"
            ]
        ):
            raise ValueError(
                "Frozen threshold identity changed "
                "inside paired metric computation."
            )

        if (
            clean_values[
                "threshold_value"
            ]
            != degraded_values[
                "threshold_value"
            ]
        ):
            raise ValueError(
                "Frozen threshold value changed "
                "inside paired metric computation."
            )

        result = {
            "schema_version": 1,
            "metric_type": (
                "paired_degradation_delta"
            ),

            "detector": (
                group[
                    "detector"
                ]
            ),

            "checkpoint_sha256": (
                group[
                    "checkpoint_sha256"
                ]
            ),

            "dataset": (
                group[
                    "dataset"
                ]
            ),

            "role": (
                group[
                    "role"
                ]
            ),

            "group_name": (
                group[
                    "group_name"
                ]
            ),

            "group_type": (
                group[
                    "group_type"
                ]
            ),

            "clean_condition": (
                group[
                    "clean_condition"
                ]
            ),

            "degraded_condition": (
                group[
                    "degraded_condition"
                ]
            ),

            "threshold_id": (
                clean_values[
                    "threshold_id"
                ]
            ),

            "threshold_value": (
                clean_values[
                    "threshold_value"
                ]
            ),

            "n_real_clean_valid": (
                group[
                    "n_real_clean_valid"
                ]
            ),

            "n_real_degraded_valid": (
                group[
                    "n_real_degraded_valid"
                ]
            ),

            "n_real_paired": (
                group[
                    "n_real_paired"
                ]
            ),

            "n_manipulated_clean_valid": (
                group[
                    "n_manipulated_clean_valid"
                ]
            ),

            "n_manipulated_degraded_valid": (
                group[
                    "n_manipulated_degraded_valid"
                ]
            ),

            "n_manipulated_paired": (
                group[
                    "n_manipulated_paired"
                ]
            ),

            "clean_auc": (
                clean_values[
                    "auc"
                ]
            ),

            "degraded_auc": (
                degraded_values[
                    "auc"
                ]
            ),

            "delta_auc": (
                degraded_values[
                    "auc"
                ]
                - clean_values[
                    "auc"
                ]
            ),

            "clean_fpr": (
                clean_values[
                    "fpr"
                ]
            ),

            "degraded_fpr": (
                degraded_values[
                    "fpr"
                ]
            ),

            "delta_fpr": (
                degraded_values[
                    "fpr"
                ]
                - clean_values[
                    "fpr"
                ]
            ),

            "clean_fnr": (
                clean_values[
                    "fnr"
                ]
            ),

            "degraded_fnr": (
                degraded_values[
                    "fnr"
                ]
            ),

            "delta_fnr": (
                degraded_values[
                    "fnr"
                ]
                - clean_values[
                    "fnr"
                ]
            ),

            "clean_hter": (
                clean_values[
                    "hter"
                ]
            ),

            "degraded_hter": (
                degraded_values[
                    "hter"
                ]
            ),

            "delta_hter": (
                degraded_values[
                    "hter"
                ]
                - clean_values[
                    "hter"
                ]
            ),

            "clean_tp": (
                clean_values[
                    "tp"
                ]
            ),

            "clean_fp": (
                clean_values[
                    "fp"
                ]
            ),

            "clean_tn": (
                clean_values[
                    "tn"
                ]
            ),

            "clean_fn": (
                clean_values[
                    "fn"
                ]
            ),

            "degraded_tp": (
                degraded_values[
                    "tp"
                ]
            ),

            "degraded_fp": (
                degraded_values[
                    "fp"
                ]
            ),

            "degraded_tn": (
                degraded_values[
                    "tn"
                ]
            ),

            "degraded_fn": (
                degraded_values[
                    "fn"
                ]
            ),
        }

        output.append(
            result
        )

    return output