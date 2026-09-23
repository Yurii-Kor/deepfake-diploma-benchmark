from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from study.analysis.analysis_schema import (
    CELEB_DATASET,
    FFPP_DATASET,
)
from study.analysis.bootstrap_metrics import (
    bootstrap_diagnostic_threshold_displacement,
    derive_stream_seed,
    percentile_interval,
)
from study.analysis.bootstrap_sampling import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    bootstrap_stratum,
    bootstrap_unit_id,
    sample_bootstrap_replicate,
)
from study.analysis.bootstrap_views import (
    build_ffpp_absolute_replicate_plan,
    build_ffpp_paired_replicate_plan,
    materialize_ffpp_absolute_group,
    materialize_ffpp_paired_group,
)
from study.analysis.build_analysis_records import (
    read_jsonl,
)
from study.analysis.metric_groups import (
    build_primary_metric_groups,
)
from study.analysis.metrics import (
    binary_auc,
    fixed_threshold_metrics,
)
from study.analysis.operating_point import (
    PRIMARY_TARGET_FPR,
    normalize_score,
    select_primary_low_fpr_threshold,
)
from study.analysis.paired_metric_groups import (
    build_paired_metric_groups,
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def read_json(
    path: Path,
) -> Dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def write_json(
    path: Path,
    value: Dict[str, Any],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            value,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write(
            "\n"
        )

    temporary.replace(
        path
    )


def _label(
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


def _score(
    row: Dict[str, Any],
) -> float:
    return normalize_score(
        row.get(
            "video_score"
        )
    )


def _valid_score_row(
    row: Dict[str, Any],
) -> bool:
    return (
        row.get(
            "analysis_eligible"
        )
        is True
        and row.get(
            "video_score"
        )
        is not None
    )


def _sampling_key(
    row: Dict[str, Any],
) -> Tuple[str, str]:
    return (
        bootstrap_stratum(
            row
        ),
        bootstrap_unit_id(
            row
        ),
    )


def _index_sampling_rows(
    rows: Iterable[
        Dict[str, Any]
    ],
    *,
    context: str,
) -> Dict[
    Tuple[str, str],
    Dict[str, Any],
]:
    index = {}

    for row in rows:
        if not _valid_score_row(
            row
        ):
            raise ValueError(
                "Invalid score row admitted to {}."
                .format(
                    context
                )
            )

        key = _sampling_key(
            row
        )

        if key in index:
            raise ValueError(
                "Duplicate bootstrap sampling key "
                "in {}: {}"
                .format(
                    context,
                    key,
                )
            )

        index[
            key
        ] = row

    return index


def _materialize_sample(
    *,
    index: Dict[
        Tuple[str, str],
        Dict[str, Any],
    ],
    sampled_units: Dict[
        str,
        List[str],
    ],
    context: str,
) -> List[Dict[str, Any]]:
    output = []

    for stratum in sorted(
        sampled_units
    ):
        for unit_id in (
            sampled_units[
                stratum
            ]
        ):
            key = (
                stratum,
                unit_id,
            )

            row = index.get(
                key
            )

            if row is None:
                raise ValueError(
                    "Sampled unit missing from {}: {}"
                    .format(
                        context,
                        key,
                    )
                )

            output.append(
                row
            )

    return output


def _metric_values(
    *,
    real_rows: List[
        Dict[str, Any]
    ],
    manipulated_rows: List[
        Dict[str, Any]
    ],
    threshold: float,
) -> Dict[str, float]:
    if not real_rows:
        raise ValueError(
            "Bootstrap metric requires real rows."
        )

    if not manipulated_rows:
        raise ValueError(
            "Bootstrap metric requires "
            "manipulated rows."
        )

    real_scores = [
        _score(
            row
        )
        for row in real_rows
    ]

    manipulated_scores = [
        _score(
            row
        )
        for row in manipulated_rows
    ]

    auc = binary_auc(
        real_scores=(
            real_scores
        ),
        manipulated_scores=(
            manipulated_scores
        ),
    )

    labels = (
        [
            0
            for _ in real_rows
        ]
        + [
            1
            for _ in manipulated_rows
        ]
    )

    decisions = (
        [
            (
                1
                if score >= threshold
                else 0
            )
            for score in real_scores
        ]
        + [
            (
                1
                if score >= threshold
                else 0
            )
            for score in (
                manipulated_scores
            )
        ]
    )

    fixed = fixed_threshold_metrics(
        labels=labels,
        decisions=decisions,
    )

    return {
        "auc": (
            auc
        ),

        "fpr": (
            fixed[
                "fpr"
            ]
        ),

        "fnr": (
            fixed[
                "fnr"
            ]
        ),

        "hter": (
            fixed[
                "hter"
            ]
        ),
    }


def _absolute_key(
    row: Dict[str, Any],
) -> Tuple[
    str,
    str,
    str,
    str,
    str,
]:
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
                "role"
            ]
        ),
        str(
            row[
                "condition"
            ]
        ),
        str(
            row[
                "group_name"
            ]
        ),
    )


def _paired_key(
    row: Dict[str, Any],
) -> Tuple[
    str,
    str,
    str,
    str,
    str,
]:
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
                "role"
            ]
        ),
        str(
            row[
                "group_name"
            ]
        ),
        str(
            row[
                "degraded_condition"
            ]
        ),
    )


def _diagnostic_key(
    row: Dict[str, Any],
) -> Tuple[str, str]:
    return (
        str(
            row[
                "detector"
            ]
        ),
        str(
            row[
                "condition"
            ]
        ),
    )


def _unique_index(
    rows: Iterable[
        Dict[str, Any]
    ],
    *,
    key_function,
    context: str,
):
    index = {}

    for row in rows:
        key = key_function(
            row
        )

        if key in index:
            raise ValueError(
                "Duplicate {} key: {}"
                .format(
                    context,
                    key,
                )
            )

        index[
            key
        ] = row

    return index


def _metric_result_indexes(
    artifact: Dict[str, Any],
):
    if (
        artifact.get(
            "artifact_type"
        )
        != "metric_results"
    ):
        raise ValueError(
            "Unexpected metric-result artifact type."
        )

    absolute = _unique_index(
        artifact.get(
            "absolute_metrics",
            [],
        ),
        key_function=(
            _absolute_key
        ),
        context=(
            "absolute metric"
        ),
    )

    paired = _unique_index(
        artifact.get(
            "paired_degradation_metrics",
            [],
        ),
        key_function=(
            _paired_key
        ),
        context=(
            "paired metric"
        ),
    )

    diagnostic = _unique_index(
        artifact.get(
            "diagnostic_thresholds",
            [],
        ),
        key_function=(
            _diagnostic_key
        ),
        context=(
            "diagnostic threshold"
        ),
    )

    return (
        absolute,
        paired,
        diagnostic,
    )


def _clean_validation_by_detector(
    analysis_records: Iterable[
        Dict[str, Any]
    ],
) -> Dict[
    str,
    List[Dict[str, Any]],
]:
    grouped = defaultdict(
        list
    )

    for row in analysis_records:
        if (
            row.get(
                "dataset"
            )
            != FFPP_DATASET
            or row.get(
                "role"
            )
            != "validation"
            or row.get(
                "condition"
            )
            != "CLN"
            or _label(
                row
            )
            != 0
            or not _valid_score_row(
                row
            )
        ):
            continue

        detector = str(
            row[
                "detector"
            ]
        )

        grouped[
            detector
        ].append(
            row
        )

    output = {}

    for detector, rows in (
        grouped.items()
    ):
        checkpoints = {
            str(
                row[
                    "checkpoint_sha256"
                ]
            )
            for row in rows
        }

        if len(
            checkpoints
        ) != 1:
            raise ValueError(
                "Detector {} has multiple "
                "checkpoints in clean validation."
                .format(
                    detector
                )
            )

        unit_ids = [
            bootstrap_unit_id(
                row
            )
            for row in rows
        ]

        if (
            len(
                unit_ids
            )
            != len(
                set(
                    unit_ids
                )
            )
        ):
            raise ValueError(
                "Duplicate clean validation "
                "bootstrap unit for detector {}."
                .format(
                    detector
                )
            )

        output[
            detector
        ] = sorted(
            rows,
            key=lambda row: (
                bootstrap_unit_id(
                    row
                )
            ),
        )

    return output


def _bootstrap_clean_thresholds(
    *,
    rows: List[
        Dict[str, Any]
    ],
    replicates: int,
    seed: int,
) -> List[float]:
    if not rows:
        raise ValueError(
            "Clean validation population is empty."
        )

    index = _index_sampling_rows(
        rows,
        context=(
            "clean validation bootstrap"
        ),
    )

    stream_seed = derive_stream_seed(
        seed=seed,
        stream_name=(
            "primary_clean_validation"
        ),
    )

    thresholds = []

    for replicate_index in range(
        replicates
    ):
        sampled_units = (
            sample_bootstrap_replicate(
                rows,
                replicate_index=(
                    replicate_index
                ),
                seed=(
                    stream_seed
                ),
            )
        )

        sampled_rows = _materialize_sample(
            index=index,
            sampled_units=(
                sampled_units
            ),
            context=(
                "clean validation bootstrap"
            ),
        )

        selection = (
            select_primary_low_fpr_threshold(
                real_scores=[
                    _score(
                        row
                    )
                    for row in sampled_rows
                ],
                target_fpr=(
                    PRIMARY_TARGET_FPR
                ),
            )
        )

        thresholds.append(
            float(
                selection[
                    "threshold_value"
                ]
            )
        )

    return thresholds


def _split_by_label(
    rows: Iterable[
        Dict[str, Any]
    ],
):
    real = []
    manipulated = []

    for row in rows:
        if _label(
            row
        ) == 0:
            real.append(
                row
            )
        else:
            manipulated.append(
                row
            )

    return (
        real,
        manipulated,
    )


def _generic_absolute_sample(
    *,
    group: Dict[str, Any],
    replicate_index: int,
    seed: int,
    stream_name: str,
):
    population = (
        list(
            group[
                "valid_real_records"
            ]
        )
        + list(
            group[
                "valid_manipulated_records"
            ]
        )
    )

    index = _index_sampling_rows(
        population,
        context=(
            stream_name
        ),
    )

    stream_seed = derive_stream_seed(
        seed=seed,
        stream_name=(
            stream_name
        ),
    )

    sampled_units = (
        sample_bootstrap_replicate(
            population,
            replicate_index=(
                replicate_index
            ),
            seed=(
                stream_seed
            ),
        )
    )

    sampled = _materialize_sample(
        index=index,
        sampled_units=(
            sampled_units
        ),
        context=(
            stream_name
        ),
    )

    return _split_by_label(
        sampled
    )


def _generic_paired_sample(
    *,
    group: Dict[str, Any],
    replicate_index: int,
    seed: int,
    stream_name: str,
):
    clean_population = (
        list(
            group[
                "clean_real_records"
            ]
        )
        + list(
            group[
                "clean_manipulated_records"
            ]
        )
    )

    degraded_population = (
        list(
            group[
                "degraded_real_records"
            ]
        )
        + list(
            group[
                "degraded_manipulated_records"
            ]
        )
    )

    clean_index = _index_sampling_rows(
        clean_population,
        context=(
            "{} clean".format(
                stream_name
            )
        ),
    )

    degraded_index = (
        _index_sampling_rows(
            degraded_population,
            context=(
                "{} degraded".format(
                    stream_name
                )
            ),
        )
    )

    if (
        set(
            clean_index
        )
        != set(
            degraded_index
        )
    ):
        raise ValueError(
            "Paired bootstrap populations differ "
            "for {}."
            .format(
                stream_name
            )
        )

    stream_seed = derive_stream_seed(
        seed=seed,
        stream_name=(
            stream_name
        ),
    )

    sampled_units = (
        sample_bootstrap_replicate(
            clean_population,
            replicate_index=(
                replicate_index
            ),
            seed=(
                stream_seed
            ),
        )
    )

    clean_sample = _materialize_sample(
        index=clean_index,
        sampled_units=(
            sampled_units
        ),
        context=(
            "{} clean".format(
                stream_name
            )
        ),
    )

    degraded_sample = _materialize_sample(
        index=degraded_index,
        sampled_units=(
            sampled_units
        ),
        context=(
            "{} degraded".format(
                stream_name
            )
        ),
    )

    (
        clean_real,
        clean_manipulated,
    ) = _split_by_label(
        clean_sample
    )

    (
        degraded_real,
        degraded_manipulated,
    ) = _split_by_label(
        degraded_sample
    )

    return {
        "clean_real": (
            clean_real
        ),

        "clean_manipulated": (
            clean_manipulated
        ),

        "degraded_real": (
            degraded_real
        ),

        "degraded_manipulated": (
            degraded_manipulated
        ),
    }


def _absolute_bootstrap(
    *,
    decision_records: List[
        Dict[str, Any]
    ],
    point_index,
    threshold_values_by_detector,
    replicates: int,
    seed: int,
):
    all_groups = (
        build_primary_metric_groups(
            decision_records
        )
    )

    groups = [
        group
        for group in all_groups
        if (
            _absolute_key(
                group
            )
            in point_index
        )
    ]

    contexts = defaultdict(
        list
    )

    for group in groups:
        key = (
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
                "condition"
            ],
        )

        contexts[
            key
        ].append(
            group
        )

    values_by_key = {}

    for group in groups:
        values_by_key[
            _absolute_key(
                group
            )
        ] = {
            "auc": [],
            "fpr": [],
            "fnr": [],
            "hter": [],
        }

    for (
        detector,
        dataset,
        role,
        condition,
    ), context_groups in sorted(
        contexts.items()
    ):
        thresholds = (
            threshold_values_by_detector[
                detector
            ]
        )

        stream_name = (
            "absolute::{}::{}::{}::{}"
            .format(
                detector,
                dataset,
                role,
                condition,
            )
        )

        for replicate_index in range(
            replicates
        ):
            threshold = thresholds[
                replicate_index
            ]

            if dataset == FFPP_DATASET:
                plan = (
                    build_ffpp_absolute_replicate_plan(
                        context_groups,
                        replicate_index=(
                            replicate_index
                        ),
                        seed=seed,
                        stream_name=(
                            stream_name
                        ),
                    )
                )

                for group in context_groups:
                    (
                        real_rows,
                        manipulated_rows,
                    ) = (
                        materialize_ffpp_absolute_group(
                            group=group,
                            replicate_plan=(
                                plan
                            ),
                        )
                    )

                    metric_values = (
                        _metric_values(
                            real_rows=(
                                real_rows
                            ),
                            manipulated_rows=(
                                manipulated_rows
                            ),
                            threshold=(
                                threshold
                            ),
                        )
                    )

                    target = (
                        values_by_key[
                            _absolute_key(
                                group
                            )
                        ]
                    )

                    for metric in (
                        "auc",
                        "fpr",
                        "fnr",
                        "hter",
                    ):
                        target[
                            metric
                        ].append(
                            metric_values[
                                metric
                            ]
                        )

            elif dataset == CELEB_DATASET:
                if len(
                    context_groups
                ) != 1:
                    raise ValueError(
                        "Celeb absolute context must "
                        "contain one pooled group."
                    )

                group = context_groups[
                    0
                ]

                (
                    real_rows,
                    manipulated_rows,
                ) = _generic_absolute_sample(
                    group=group,
                    replicate_index=(
                        replicate_index
                    ),
                    seed=seed,
                    stream_name=(
                        stream_name
                    ),
                )

                metric_values = (
                    _metric_values(
                        real_rows=(
                            real_rows
                        ),
                        manipulated_rows=(
                            manipulated_rows
                        ),
                        threshold=(
                            threshold
                        ),
                    )
                )

                target = (
                    values_by_key[
                        _absolute_key(
                            group
                        )
                    ]
                )

                for metric in (
                    "auc",
                    "fpr",
                    "fnr",
                    "hter",
                ):
                    target[
                        metric
                    ].append(
                        metric_values[
                            metric
                        ]
                    )

            else:
                raise ValueError(
                    "Unsupported absolute bootstrap "
                    "dataset: {}"
                    .format(
                        dataset
                    )
                )

    if (
        set(
            values_by_key
        )
        != set(
            point_index
        )
    ):
        raise ValueError(
            "Absolute bootstrap keys do not match "
            "5.6.3 point-result keys."
        )

    output = []

    for key in sorted(
        values_by_key
    ):
        point = point_index[
            key
        ]

        values = values_by_key[
            key
        ]

        output.append(
            {
                "detector": (
                    point[
                        "detector"
                    ]
                ),

                "checkpoint_sha256": (
                    point[
                        "checkpoint_sha256"
                    ]
                ),

                "dataset": (
                    point[
                        "dataset"
                    ]
                ),

                "role": (
                    point[
                        "role"
                    ]
                ),

                "condition": (
                    point[
                        "condition"
                    ]
                ),

                "group_name": (
                    point[
                        "group_name"
                    ]
                ),

                "group_type": (
                    point[
                        "group_type"
                    ]
                ),

                "point_estimate": {
                    metric: float(
                        point[
                            metric
                        ]
                    )
                    for metric in (
                        "auc",
                        "fpr",
                        "fnr",
                        "hter",
                    )
                },

                "confidence_intervals": {
                    metric: (
                        percentile_interval(
                            values[
                                metric
                            ],
                            ci_level=(
                                BOOTSTRAP_CI_LEVEL
                            ),
                        )
                    )
                    for metric in (
                        "auc",
                        "fpr",
                        "fnr",
                        "hter",
                    )
                },

                "bootstrap_values": (
                    values
                ),
            }
        )

    return output


def _paired_bootstrap(
    *,
    decision_records: List[
        Dict[str, Any]
    ],
    point_index,
    threshold_values_by_detector,
    replicates: int,
    seed: int,
):
    all_groups = (
        build_paired_metric_groups(
            decision_records
        )
    )

    groups = [
        group
        for group in all_groups
        if (
            _paired_key(
                group
            )
            in point_index
        )
    ]

    contexts = defaultdict(
        list
    )

    for group in groups:
        key = (
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
                "degraded_condition"
            ],
        )

        contexts[
            key
        ].append(
            group
        )

    values_by_key = {}

    for group in groups:
        values_by_key[
            _paired_key(
                group
            )
        ] = {
            "delta_auc": [],
            "delta_fpr": [],
            "delta_fnr": [],
            "delta_hter": [],
        }

    for (
        detector,
        dataset,
        role,
        degraded_condition,
    ), context_groups in sorted(
        contexts.items()
    ):
        thresholds = (
            threshold_values_by_detector[
                detector
            ]
        )

        stream_name = (
            "paired::{}::{}::{}::{}"
            .format(
                detector,
                dataset,
                role,
                degraded_condition,
            )
        )

        for replicate_index in range(
            replicates
        ):
            threshold = thresholds[
                replicate_index
            ]

            if dataset == FFPP_DATASET:
                plan = (
                    build_ffpp_paired_replicate_plan(
                        context_groups,
                        replicate_index=(
                            replicate_index
                        ),
                        seed=seed,
                        stream_name=(
                            stream_name
                        ),
                    )
                )

                for group in context_groups:
                    sampled = (
                        materialize_ffpp_paired_group(
                            group=group,
                            replicate_plan=(
                                plan
                            ),
                        )
                    )

                    clean_values = (
                        _metric_values(
                            real_rows=(
                                sampled[
                                    "clean_real"
                                ]
                            ),
                            manipulated_rows=(
                                sampled[
                                    "clean_manipulated"
                                ]
                            ),
                            threshold=(
                                threshold
                            ),
                        )
                    )

                    degraded_values = (
                        _metric_values(
                            real_rows=(
                                sampled[
                                    "degraded_real"
                                ]
                            ),
                            manipulated_rows=(
                                sampled[
                                    "degraded_manipulated"
                                ]
                            ),
                            threshold=(
                                threshold
                            ),
                        )
                    )

                    target = (
                        values_by_key[
                            _paired_key(
                                group
                            )
                        ]
                    )

                    for metric in (
                        "auc",
                        "fpr",
                        "fnr",
                        "hter",
                    ):
                        target[
                            "delta_{}".format(
                                metric
                            )
                        ].append(
                            degraded_values[
                                metric
                            ]
                            - clean_values[
                                metric
                            ]
                        )

            elif dataset == CELEB_DATASET:
                if len(
                    context_groups
                ) != 1:
                    raise ValueError(
                        "Celeb paired context must "
                        "contain one pooled group."
                    )

                group = context_groups[
                    0
                ]

                sampled = (
                    _generic_paired_sample(
                        group=group,
                        replicate_index=(
                            replicate_index
                        ),
                        seed=seed,
                        stream_name=(
                            stream_name
                        ),
                    )
                )

                clean_values = (
                    _metric_values(
                        real_rows=(
                            sampled[
                                "clean_real"
                            ]
                        ),
                        manipulated_rows=(
                            sampled[
                                "clean_manipulated"
                            ]
                        ),
                        threshold=(
                            threshold
                        ),
                    )
                )

                degraded_values = (
                    _metric_values(
                        real_rows=(
                            sampled[
                                "degraded_real"
                            ]
                        ),
                        manipulated_rows=(
                            sampled[
                                "degraded_manipulated"
                            ]
                        ),
                        threshold=(
                            threshold
                        ),
                    )
                )

                target = (
                    values_by_key[
                        _paired_key(
                            group
                        )
                    ]
                )

                for metric in (
                    "auc",
                    "fpr",
                    "fnr",
                    "hter",
                ):
                    target[
                        "delta_{}".format(
                            metric
                        )
                    ].append(
                        degraded_values[
                            metric
                        ]
                        - clean_values[
                            metric
                        ]
                    )

            else:
                raise ValueError(
                    "Unsupported paired bootstrap "
                    "dataset: {}"
                    .format(
                        dataset
                    )
                )

    if (
        set(
            values_by_key
        )
        != set(
            point_index
        )
    ):
        raise ValueError(
            "Paired bootstrap keys do not match "
            "5.6.3 point-result keys."
        )

    output = []

    for key in sorted(
        values_by_key
    ):
        point = point_index[
            key
        ]

        values = values_by_key[
            key
        ]

        output.append(
            {
                "detector": (
                    point[
                        "detector"
                    ]
                ),

                "checkpoint_sha256": (
                    point[
                        "checkpoint_sha256"
                    ]
                ),

                "dataset": (
                    point[
                        "dataset"
                    ]
                ),

                "role": (
                    point[
                        "role"
                    ]
                ),

                "group_name": (
                    point[
                        "group_name"
                    ]
                ),

                "group_type": (
                    point[
                        "group_type"
                    ]
                ),

                "clean_condition": (
                    point[
                        "clean_condition"
                    ]
                ),

                "degraded_condition": (
                    point[
                        "degraded_condition"
                    ]
                ),

                "point_estimate": {
                    metric: float(
                        point[
                            metric
                        ]
                    )
                    for metric in (
                        "delta_auc",
                        "delta_fpr",
                        "delta_fnr",
                        "delta_hter",
                    )
                },

                "confidence_intervals": {
                    metric: (
                        percentile_interval(
                            values[
                                metric
                            ],
                            ci_level=(
                                BOOTSTRAP_CI_LEVEL
                            ),
                        )
                    )
                    for metric in (
                        "delta_auc",
                        "delta_fpr",
                        "delta_fnr",
                        "delta_hter",
                    )
                },

                "bootstrap_values": (
                    values
                ),
            }
        )

    return output


def _diagnostic_bootstrap(
    *,
    analysis_records: List[
        Dict[str, Any]
    ],
    point_index,
    replicates: int,
    seed: int,
):
    output = []

    for (
        detector,
        condition,
    ), point in sorted(
        point_index.items()
    ):
        detector_rows = [
            row
            for row in analysis_records
            if (
                row.get(
                    "detector"
                )
                == detector
            )
        ]

        clean_rows = [
            row
            for row in detector_rows
            if (
                row.get(
                    "dataset"
                )
                == FFPP_DATASET
                and row.get(
                    "role"
                )
                == "validation"
                and row.get(
                    "condition"
                )
                == "CLN"
                and _label(
                    row
                )
                == 0
            )
        ]

        degraded_rows = [
            row
            for row in detector_rows
            if (
                row.get(
                    "dataset"
                )
                == FFPP_DATASET
                and row.get(
                    "role"
                )
                == "validation"
                and row.get(
                    "condition"
                )
                == condition
                and _label(
                    row
                )
                == 0
            )
        ]

        result = (
            bootstrap_diagnostic_threshold_displacement(
                clean_validation_rows=(
                    clean_rows
                ),
                degraded_validation_rows=(
                    degraded_rows
                ),
                degraded_condition=(
                    condition
                ),
                replicates=(
                    replicates
                ),
                seed=seed,
            )
        )

        clean_values = [
            float(
                row[
                    "clean_threshold"
                ]
            )
            for row in (
                result[
                    "replicates"
                ]
            )
        ]

        diagnostic_values = [
            float(
                row[
                    "diagnostic_threshold"
                ]
            )
            for row in (
                result[
                    "replicates"
                ]
            )
        ]

        displacement_values = [
            float(
                row[
                    "threshold_displacement"
                ]
            )
            for row in (
                result[
                    "replicates"
                ]
            )
        ]

        output.append(
            {
                "detector": (
                    detector
                ),

                "checkpoint_sha256": (
                    point[
                        "checkpoint_sha256"
                    ]
                ),

                "dataset": (
                    FFPP_DATASET
                ),

                "role": (
                    "validation"
                ),

                "degraded_condition": (
                    condition
                ),

                "n_clean_valid_real": (
                    result[
                        "n_clean_valid_real"
                    ]
                ),

                "n_degraded_valid_real": (
                    result[
                        "n_degraded_valid_real"
                    ]
                ),

                "n_paired_valid_real": (
                    result[
                        "n_paired_valid_real"
                    ]
                ),

                "point_estimate": {
                    "threshold_displacement": (
                        float(
                            point[
                                "threshold_displacement"
                            ]
                        )
                    )
                },

                "confidence_intervals": {
                    "threshold_displacement": (
                        percentile_interval(
                            displacement_values,
                            ci_level=(
                                BOOTSTRAP_CI_LEVEL
                            ),
                        )
                    )
                },

                "bootstrap_values": {
                    "clean_threshold": (
                        clean_values
                    ),

                    "diagnostic_threshold": (
                        diagnostic_values
                    ),

                    "threshold_displacement": (
                        displacement_values
                    ),
                },
            }
        )

    return output


def build_bootstrap_result_artifact(
    *,
    analysis_records: List[
        Dict[str, Any]
    ],
    decision_records: List[
        Dict[str, Any]
    ],
    metric_result_artifact: Dict[
        str,
        Any,
    ],
    replicates: int = BOOTSTRAP_REPLICATES,
    allow_empty_primary_metrics: bool = False,
    allow_nonproduction_replicates: bool = False,
) -> Dict[str, Any]:
    if replicates <= 0:
        raise ValueError(
            "replicates must be positive."
        )

    if (
        replicates
        != BOOTSTRAP_REPLICATES
        and not allow_nonproduction_replicates
    ):
        raise ValueError(
            "Production bootstrap is frozen at "
            "{} replicates."
            .format(
                BOOTSTRAP_REPLICATES
            )
        )

    (
        absolute_point_index,
        paired_point_index,
        diagnostic_point_index,
    ) = _metric_result_indexes(
        metric_result_artifact
    )

    if (
        not allow_empty_primary_metrics
        and not absolute_point_index
    ):
        raise RuntimeError(
            "No held-out absolute metric results "
            "are available."
        )

    if (
        not allow_empty_primary_metrics
        and not paired_point_index
    ):
        raise RuntimeError(
            "No paired degradation results "
            "are available."
        )

    if not diagnostic_point_index:
        raise RuntimeError(
            "No diagnostic threshold results "
            "are available."
        )

    calibration_by_detector = (
        _clean_validation_by_detector(
            analysis_records
        )
    )

    expected_detectors = sorted(
        {
            str(
                detector
            )
            for detector in (
                metric_result_artifact.get(
                    "detectors",
                    [],
                )
            )
        }
    )

    if not expected_detectors:
        raise ValueError(
            "Metric-result artifact has no detectors."
        )

    threshold_values_by_detector = {}
    threshold_bootstrap = []

    for detector in expected_detectors:
        calibration_rows = (
            calibration_by_detector.get(
                detector
            )
        )

        if not calibration_rows:
            raise ValueError(
                "No valid clean validation "
                "calibration rows for detector {}."
                .format(
                    detector
                )
            )

        thresholds = (
            _bootstrap_clean_thresholds(
                rows=(
                    calibration_rows
                ),
                replicates=(
                    replicates
                ),
                seed=(
                    BOOTSTRAP_SEED
                ),
            )
        )

        threshold_values_by_detector[
            detector
        ] = thresholds

        checkpoints = {
            str(
                row[
                    "checkpoint_sha256"
                ]
            )
            for row in (
                calibration_rows
            )
        }

        checkpoint_sha256 = next(
            iter(
                checkpoints
            )
        )

        threshold_bootstrap.append(
            {
                "detector": (
                    detector
                ),

                "checkpoint_sha256": (
                    checkpoint_sha256
                ),

                "valid_clean_validation_real": (
                    len(
                        calibration_rows
                    )
                ),

                "provenance_only": True,

                "bootstrap_values": (
                    thresholds
                ),

                "percentile_interval": (
                    percentile_interval(
                        thresholds,
                        ci_level=(
                            BOOTSTRAP_CI_LEVEL
                        ),
                    )
                ),
            }
        )

    absolute_results = (
        _absolute_bootstrap(
            decision_records=(
                decision_records
            ),
            point_index=(
                absolute_point_index
            ),
            threshold_values_by_detector=(
                threshold_values_by_detector
            ),
            replicates=(
                replicates
            ),
            seed=(
                BOOTSTRAP_SEED
            ),
        )
    )

    paired_results = (
        _paired_bootstrap(
            decision_records=(
                decision_records
            ),
            point_index=(
                paired_point_index
            ),
            threshold_values_by_detector=(
                threshold_values_by_detector
            ),
            replicates=(
                replicates
            ),
            seed=(
                BOOTSTRAP_SEED
            ),
        )
    )

    diagnostic_results = (
        _diagnostic_bootstrap(
            analysis_records=(
                analysis_records
            ),
            point_index=(
                diagnostic_point_index
            ),
            replicates=(
                replicates
            ),
            seed=(
                BOOTSTRAP_SEED
            ),
        )
    )

    return {
        "schema_version": 1,

        "artifact_type": (
            "bootstrap_results"
        ),

        "bootstrap_configuration": {
            "replicates": (
                replicates
            ),

            "ci_level": (
                BOOTSTRAP_CI_LEVEL
            ),

            "ci_method": (
                "percentile_linear_interpolation"
            ),

            "seed": (
                BOOTSTRAP_SEED
            ),

            "rng_stream_policy": (
                "deterministic_named_streams"
            ),

            "primary_target_fpr": (
                PRIMARY_TARGET_FPR
            ),

            "threshold_reestimated_each_replicate": (
                True
            ),

            "paired_clean_degraded_sampling": (
                True
            ),

            "production_configuration": (
                replicates
                == BOOTSTRAP_REPLICATES
            ),
        },

        "detectors": (
            expected_detectors
        ),

        "clean_threshold_bootstrap": (
            threshold_bootstrap
        ),

        "absolute_metric_bootstrap_count": (
            len(
                absolute_results
            )
        ),

        "paired_degradation_bootstrap_count": (
            len(
                paired_results
            )
        ),

        "diagnostic_threshold_bootstrap_count": (
            len(
                diagnostic_results
            )
        ),

        "absolute_metrics": (
            absolute_results
        ),

        "paired_degradation_metrics": (
            paired_results
        ),

        "diagnostic_thresholds": (
            diagnostic_results
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the controlled paired-bootstrap "
            "result artifact for Section 5.6.4."
        )
    )

    parser.add_argument(
        "--analysis-jsonl",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--decisions-jsonl",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--metric-results-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--summary-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=(
            BOOTSTRAP_REPLICATES
        ),
    )

    parser.add_argument(
        "--allow-empty-primary-metrics",
        action="store_true",
        help=(
            "Permit validation-only smoke input "
            "without held-out absolute or paired "
            "evaluation results."
        ),
    )

    parser.add_argument(
        "--allow-nonproduction-bootstrap",
        action="store_true",
        help=(
            "Permit a replicate count other than "
            "the frozen production value. Intended "
            "only for controlled validation."
        ),
    )

    args = parser.parse_args()

    analysis_records = read_jsonl(
        args.analysis_jsonl
    )

    decision_records = read_jsonl(
        args.decisions_jsonl
    )

    metric_result_artifact = read_json(
        args.metric_results_json
    )

    artifact = (
        build_bootstrap_result_artifact(
            analysis_records=(
                analysis_records
            ),
            decision_records=(
                decision_records
            ),
            metric_result_artifact=(
                metric_result_artifact
            ),
            replicates=(
                args.replicates
            ),
            allow_empty_primary_metrics=(
                args.allow_empty_primary_metrics
            ),
            allow_nonproduction_replicates=(
                args.allow_nonproduction_bootstrap
            ),
        )
    )

    artifact[
        "source_artifacts"
    ] = {
        "analysis_jsonl": {
            "path": str(
                args.analysis_jsonl
            ),

            "sha256": (
                sha256_file(
                    args.analysis_jsonl
                )
            ),
        },

        "decisions_jsonl": {
            "path": str(
                args.decisions_jsonl
            ),

            "sha256": (
                sha256_file(
                    args.decisions_jsonl
                )
            ),
        },

        "metric_results_json": {
            "path": str(
                args.metric_results_json
            ),

            "sha256": (
                sha256_file(
                    args.metric_results_json
                )
            ),
        },
    }

    write_json(
        args.output_json,
        artifact,
    )

    summary = {
        "schema_version": 1,

        "artifact_type": (
            "bootstrap_results_summary"
        ),

        "bootstrap_configuration": (
            artifact[
                "bootstrap_configuration"
            ]
        ),

        "detectors": (
            artifact[
                "detectors"
            ]
        ),

        "absolute_metric_bootstrap_count": (
            artifact[
                "absolute_metric_bootstrap_count"
            ]
        ),

        "paired_degradation_bootstrap_count": (
            artifact[
                "paired_degradation_bootstrap_count"
            ]
        ),

        "diagnostic_threshold_bootstrap_count": (
            artifact[
                "diagnostic_threshold_bootstrap_count"
            ]
        ),

        "output": {
            "path": str(
                args.output_json
            ),

            "sha256": (
                sha256_file(
                    args.output_json
                )
            ),
        },

        "status": "passed",
    }

    write_json(
        args.summary_json,
        summary,
    )

    print(
        "BOOTSTRAP RESULT ARTIFACT BUILD"
    )
    print(
        "  replicates:                  {}"
        .format(
            artifact[
                "bootstrap_configuration"
            ][
                "replicates"
            ]
        )
    )
    print(
        "  CI level:                    {}"
        .format(
            artifact[
                "bootstrap_configuration"
            ][
                "ci_level"
            ]
        )
    )
    print(
        "  seed:                        {}"
        .format(
            artifact[
                "bootstrap_configuration"
            ][
                "seed"
            ]
        )
    )
    print(
        "  absolute metric results:     {}"
        .format(
            artifact[
                "absolute_metric_bootstrap_count"
            ]
        )
    )
    print(
        "  paired degradation results:  {}"
        .format(
            artifact[
                "paired_degradation_bootstrap_count"
            ]
        )
    )
    print(
        "  diagnostic threshold results:{}"
        .format(
            artifact[
                "diagnostic_threshold_bootstrap_count"
            ]
        )
    )
    print(
        "  output:                      {}"
        .format(
            args.output_json
        )
    )
    print(
        "  summary:                     {}"
        .format(
            args.summary_json
        )
    )
    print()
    print(
        "BOOTSTRAP RESULT ARTIFACT BUILD PASSED"
    )


if __name__ == "__main__":
    main()