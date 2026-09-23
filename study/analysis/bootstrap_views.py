from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from study.analysis.bootstrap_metrics import (
    derive_stream_seed,
)
from study.analysis.bootstrap_sampling import (
    BOOTSTRAP_SEED,
    FFPP_DATASET,
    FFPP_MANIPULATIONS,
    bootstrap_stratum,
    bootstrap_unit_id,
    sample_bootstrap_replicate,
)


FFPP_GROUP_ORDER = (
    "DeepFakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
    "pooled",
)


def _index_by_unit(
    rows: Iterable[
        Dict[str, Any]
    ],
    *,
    expected_stratum: str,
    context: str,
) -> Dict[str, Dict[str, Any]]:
    index = {}

    for row in rows:
        stratum = bootstrap_stratum(
            row
        )

        if stratum != expected_stratum:
            raise ValueError(
                "{} contains stratum {!r}; "
                "expected {!r}."
                .format(
                    context,
                    stratum,
                    expected_stratum,
                )
            )

        unit_id = bootstrap_unit_id(
            row
        )

        if unit_id in index:
            raise ValueError(
                "Duplicate bootstrap unit in {}: {}"
                .format(
                    context,
                    unit_id,
                )
            )

        index[
            unit_id
        ] = row

    return index


def _materialize_units(
    *,
    index: Dict[str, Dict[str, Any]],
    sampled_units: List[str],
    context: str,
) -> List[Dict[str, Any]]:
    output = []

    for unit_id in sampled_units:
        row = index.get(
            unit_id
        )

        if row is None:
            raise ValueError(
                "Sampled unit {} is missing from {}."
                .format(
                    unit_id,
                    context,
                )
            )

        output.append(
            row
        )

    return output


def _validate_ffpp_group_contexts(
    groups: Iterable[
        Dict[str, Any]
    ],
):
    groups = list(
        groups
    )

    if not groups:
        raise ValueError(
            "FF++ bootstrap-view input is empty."
        )

    context_values = {
        (
            str(
                group[
                    "detector"
                ]
            ),
            str(
                group[
                    "checkpoint_sha256"
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
        )
        for group in groups
    }

    if len(
        context_values
    ) != 1:
        raise ValueError(
            "FF++ bootstrap views must belong to "
            "one detector/checkpoint/dataset/role "
            "context."
        )

    (
        _,
        _,
        dataset,
        _,
    ) = next(
        iter(
            context_values
        )
    )

    if dataset != FFPP_DATASET:
        raise ValueError(
            "FF++ bootstrap-view constructor "
            "received another dataset."
        )

    names = [
        str(
            group[
                "group_name"
            ]
        )
        for group in groups
    ]

    if len(
        names
    ) != len(
        set(
            names
        )
    ):
        raise ValueError(
            "Duplicate FF++ metric group."
        )


def _group_index(
    groups: Iterable[
        Dict[str, Any]
    ],
) -> Dict[str, Dict[str, Any]]:
    _validate_ffpp_group_contexts(
        groups
    )

    return {
        str(
            group[
                "group_name"
            ]
        ): group
        for group in groups
    }


def build_ffpp_absolute_replicate_plan(
    groups: Iterable[
        Dict[str, Any]
    ],
    *,
    replicate_index: int,
    seed: int = BOOTSTRAP_SEED,
    stream_name: str = (
        "ffpp_absolute_views"
    ),
) -> Dict[str, List[str]]:
    """
    Build one stratified FF++ resample for a single
    detector/role/condition.

    The returned plan contains one real draw and one
    draw for each manipulation method. Method-specific
    and pooled analyses must reuse this exact plan.
    """

    index = _group_index(
        groups
    )

    pooled = index.get(
        "pooled"
    )

    if pooled is None:
        raise ValueError(
            "FF++ pooled metric group is required "
            "to construct a shared bootstrap plan."
        )

    population = (
        list(
            pooled[
                "valid_real_records"
            ]
        )
        + list(
            pooled[
                "valid_manipulated_records"
            ]
        )
    )

    if not population:
        raise ValueError(
            "FF++ pooled valid population is empty."
        )

    stream_seed = derive_stream_seed(
        seed=seed,
        stream_name=(
            stream_name
        ),
    )

    plan = sample_bootstrap_replicate(
        population,
        replicate_index=(
            replicate_index
        ),
        seed=(
            stream_seed
        ),
    )

    expected_strata = {
        "real",
        "DeepFakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
    }

    if (
        set(
            plan
        )
        != expected_strata
    ):
        raise ValueError(
            "Unexpected FF++ bootstrap strata: {}"
            .format(
                sorted(
                    plan
                )
            )
        )

    return plan


def materialize_ffpp_absolute_group(
    *,
    group: Dict[str, Any],
    replicate_plan: Dict[
        str,
        List[str],
    ],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    group_name = str(
        group[
            "group_name"
        ]
    )

    if (
        group_name
        not in FFPP_GROUP_ORDER
    ):
        raise ValueError(
            "Unsupported FF++ group: {}"
            .format(
                group_name
            )
        )

    real_index = _index_by_unit(
        group[
            "valid_real_records"
        ],
        expected_stratum="real",
        context=(
            "{} real".format(
                group_name
            )
        ),
    )

    sampled_real = _materialize_units(
        index=real_index,
        sampled_units=(
            replicate_plan[
                "real"
            ]
        ),
        context=(
            "{} real".format(
                group_name
            )
        ),
    )

    if group_name != "pooled":
        manipulated_index = _index_by_unit(
            group[
                "valid_manipulated_records"
            ],
            expected_stratum=(
                group_name
            ),
            context=(
                "{} manipulated".format(
                    group_name
                )
            ),
        )

        sampled_manipulated = (
            _materialize_units(
                index=(
                    manipulated_index
                ),
                sampled_units=(
                    replicate_plan[
                        group_name
                    ]
                ),
                context=(
                    "{} manipulated"
                    .format(
                        group_name
                    )
                ),
            )
        )

        return (
            sampled_real,
            sampled_manipulated,
        )

    #
    # Pooled fake population is not resampled again.
    # It is constructed by concatenating the four
    # already sampled method-specific strata.
    #
    sampled_manipulated = []

    pooled_rows = list(
        group[
            "valid_manipulated_records"
        ]
    )

    for manipulation in (
        FFPP_MANIPULATIONS
    ):
        method_rows = [
            row
            for row in pooled_rows
            if (
                bootstrap_stratum(
                    row
                )
                == manipulation
            )
        ]

        method_index = _index_by_unit(
            method_rows,
            expected_stratum=(
                manipulation
            ),
            context=(
                "pooled {}".format(
                    manipulation
                )
            ),
        )

        sampled_manipulated.extend(
            _materialize_units(
                index=(
                    method_index
                ),
                sampled_units=(
                    replicate_plan[
                        manipulation
                    ]
                ),
                context=(
                    "pooled {}".format(
                        manipulation
                    )
                ),
            )
        )

    return (
        sampled_real,
        sampled_manipulated,
    )


def build_ffpp_paired_replicate_plan(
    groups: Iterable[
        Dict[str, Any]
    ],
    *,
    replicate_index: int,
    seed: int = BOOTSTRAP_SEED,
    stream_name: str = (
        "ffpp_paired_views"
    ),
) -> Dict[str, List[str]]:
    """
    Construct one FF++ plan from the paired CLN-side
    population. The same plan is later applied to both
    CLN and the degraded condition.
    """

    index = _group_index(
        groups
    )

    pooled = index.get(
        "pooled"
    )

    if pooled is None:
        raise ValueError(
            "Paired FF++ pooled group is required."
        )

    population = (
        list(
            pooled[
                "clean_real_records"
            ]
        )
        + list(
            pooled[
                "clean_manipulated_records"
            ]
        )
    )

    if not population:
        raise ValueError(
            "Paired FF++ population is empty."
        )

    stream_seed = derive_stream_seed(
        seed=seed,
        stream_name=(
            stream_name
        ),
    )

    plan = sample_bootstrap_replicate(
        population,
        replicate_index=(
            replicate_index
        ),
        seed=(
            stream_seed
        ),
    )

    expected_strata = {
        "real",
        "DeepFakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
    }

    if (
        set(
            plan
        )
        != expected_strata
    ):
        raise ValueError(
            "Unexpected paired FF++ strata."
        )

    return plan


def _materialize_paired_class(
    *,
    clean_rows: List[
        Dict[str, Any]
    ],
    degraded_rows: List[
        Dict[str, Any]
    ],
    sampled_units: List[str],
    stratum: str,
    context: str,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    clean_index = _index_by_unit(
        clean_rows,
        expected_stratum=(
            stratum
        ),
        context=(
            "{} clean".format(
                context
            )
        ),
    )

    degraded_index = _index_by_unit(
        degraded_rows,
        expected_stratum=(
            stratum
        ),
        context=(
            "{} degraded".format(
                context
            )
        ),
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
            "{} does not contain identical "
            "CLN/degraded bootstrap units."
            .format(
                context
            )
        )

    sampled_clean = _materialize_units(
        index=clean_index,
        sampled_units=(
            sampled_units
        ),
        context=(
            "{} clean".format(
                context
            )
        ),
    )

    sampled_degraded = (
        _materialize_units(
            index=degraded_index,
            sampled_units=(
                sampled_units
            ),
            context=(
                "{} degraded".format(
                    context
                )
            ),
        )
    )

    return (
        sampled_clean,
        sampled_degraded,
    )


def materialize_ffpp_paired_group(
    *,
    group: Dict[str, Any],
    replicate_plan: Dict[
        str,
        List[str],
    ],
) -> Dict[str, List[Dict[str, Any]]]:
    group_name = str(
        group[
            "group_name"
        ]
    )

    if (
        group_name
        not in FFPP_GROUP_ORDER
    ):
        raise ValueError(
            "Unsupported paired FF++ group: {}"
            .format(
                group_name
            )
        )

    (
        clean_real,
        degraded_real,
    ) = _materialize_paired_class(
        clean_rows=(
            group[
                "clean_real_records"
            ]
        ),
        degraded_rows=(
            group[
                "degraded_real_records"
            ]
        ),
        sampled_units=(
            replicate_plan[
                "real"
            ]
        ),
        stratum="real",
        context=(
            "{} real".format(
                group_name
            )
        ),
    )

    if group_name != "pooled":
        (
            clean_manipulated,
            degraded_manipulated,
        ) = _materialize_paired_class(
            clean_rows=(
                group[
                    "clean_manipulated_records"
                ]
            ),
            degraded_rows=(
                group[
                    "degraded_manipulated_records"
                ]
            ),
            sampled_units=(
                replicate_plan[
                    group_name
                ]
            ),
            stratum=(
                group_name
            ),
            context=(
                "{} manipulated".format(
                    group_name
                )
            ),
        )

    else:
        clean_manipulated = []
        degraded_manipulated = []

        clean_pool = list(
            group[
                "clean_manipulated_records"
            ]
        )

        degraded_pool = list(
            group[
                "degraded_manipulated_records"
            ]
        )

        for manipulation in (
            FFPP_MANIPULATIONS
        ):
            clean_method = [
                row
                for row in clean_pool
                if (
                    bootstrap_stratum(
                        row
                    )
                    == manipulation
                )
            ]

            degraded_method = [
                row
                for row in degraded_pool
                if (
                    bootstrap_stratum(
                        row
                    )
                    == manipulation
                )
            ]

            (
                sampled_clean_method,
                sampled_degraded_method,
            ) = _materialize_paired_class(
                clean_rows=(
                    clean_method
                ),
                degraded_rows=(
                    degraded_method
                ),
                sampled_units=(
                    replicate_plan[
                        manipulation
                    ]
                ),
                stratum=(
                    manipulation
                ),
                context=(
                    "pooled {}".format(
                        manipulation
                    )
                ),
            )

            clean_manipulated.extend(
                sampled_clean_method
            )

            degraded_manipulated.extend(
                sampled_degraded_method
            )

    return {
        "clean_real": (
            clean_real
        ),

        "degraded_real": (
            degraded_real
        ),

        "clean_manipulated": (
            clean_manipulated
        ),

        "degraded_manipulated": (
            degraded_manipulated
        ),
    }