from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from study.reproducibility.validate_analysis_artifacts import (
    analysis_record_key,
    validate_analysis_records,
)
from study.reproducibility.validate_inference_analysis_chain import (
    inference_record_key,
    validate_inference_records,
)


FFPP_DATASET = "FaceForensics++"
CELEB_DATASET = "Celeb-DF-v2"

MANIFEST_FIELDS = (
    "dataset",
    "source_split",
    "role",
    "base_video_id",
    "source_group_id",
    "source_video_id",
    "target_video_id",
    "study_label",
    "manipulation",
    "relative_source_path",
    "source_label",
)

FFPP_MANIPULATIONS = {
    "DeepFakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
}

SUPPORTED_DATASET_ROLE_SPLITS = {
    (
        FFPP_DATASET,
        "train",
        "fit",
    ),
    (
        FFPP_DATASET,
        "train",
        "development",
    ),
    (
        FFPP_DATASET,
        "val",
        "validation",
    ),
    (
        FFPP_DATASET,
        "test",
        "test",
    ),
    (
        CELEB_DATASET,
        "test",
        "external_evaluation",
    ),
}


def read_manifest_csv(
    path: Path,
) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        actual_fields = tuple(
            reader.fieldnames
            or ()
        )

        if actual_fields != MANIFEST_FIELDS:
            raise ValueError(
                "Canonical manifest schema mismatch.\n"
                "Expected: {}\n"
                "Actual:   {}".format(
                    MANIFEST_FIELDS,
                    actual_fields,
                )
            )

        rows = list(
            reader
        )

    if not rows:
        raise ValueError(
            "Canonical manifest is empty."
        )

    return rows


def required_csv_string(
    row: Dict[str, Any],
    field_name: str,
) -> str:
    if field_name not in row:
        raise ValueError(
            "Missing canonical manifest field: {}."
            .format(
                field_name
            )
        )

    value = row[
        field_name
    ]

    if type(value) is not str:
        raise ValueError(
            "{} must be a CSV string; got {!r} "
            "(type {}).".format(
                field_name,
                value,
                type(value).__name__,
            )
        )

    if value == "":
        raise ValueError(
            "{} must not be empty.".format(
                field_name
            )
        )

    return value


def optional_csv_string(
    row: Dict[str, Any],
    field_name: str,
) -> str:
    if field_name not in row:
        raise ValueError(
            "Missing canonical manifest field: {}."
            .format(
                field_name
            )
        )

    value = row[
        field_name
    ]

    if type(value) is not str:
        raise ValueError(
            "{} must be a CSV string; got {!r} "
            "(type {}).".format(
                field_name,
                value,
                type(value).__name__,
            )
        )

    return value


def strict_csv_binary_int(
    value: Any,
    *,
    field_name: str,
) -> int:
    #
    # Canonical manifest values are read from CSV,
    # therefore the only accepted representations are
    # the exact strings "0" and "1".
    #
    # Do not use int(value): values such as "0.7"
    # must fail rather than being normalized elsewhere.
    #
    if type(value) is not str:
        raise ValueError(
            "{} must be encoded as CSV string "
            "'0' or '1'; got {!r} (type {}).".format(
                field_name,
                value,
                type(value).__name__,
            )
        )

    if value not in (
        "0",
        "1",
    ):
        raise ValueError(
            "{} must be exactly '0' or '1'; "
            "got {!r}.".format(
                field_name,
                value,
            )
        )

    return (
        0
        if value == "0"
        else 1
    )


def manifest_key(
    row: Dict[str, Any],
) -> Tuple[str, str]:
    return (
        required_csv_string(
            row,
            "dataset",
        ),
        required_csv_string(
            row,
            "relative_source_path",
        ),
    )


def validate_manifest_record(
    row: Dict[str, Any],
) -> None:
    dataset = required_csv_string(
        row,
        "dataset",
    )

    source_split = required_csv_string(
        row,
        "source_split",
    )

    role = required_csv_string(
        row,
        "role",
    )

    required_csv_string(
        row,
        "base_video_id",
    )

    required_csv_string(
        row,
        "source_group_id",
    )

    required_csv_string(
        row,
        "source_video_id",
    )

    target_video_id = (
        optional_csv_string(
            row,
            "target_video_id",
        )
    )

    study_label = (
        strict_csv_binary_int(
            row.get(
                "study_label"
            ),
            field_name="study_label",
        )
    )

    manipulation = optional_csv_string(
        row,
        "manipulation",
    )

    required_csv_string(
        row,
        "relative_source_path",
    )

    optional_csv_string(
        row,
        "source_label",
    )

    identity = (
        dataset,
        source_split,
        role,
    )

    if (
        identity
        not in SUPPORTED_DATASET_ROLE_SPLITS
    ):
        raise ValueError(
            "Unsupported canonical dataset/split/role "
            "combination: {}.".format(
                identity
            )
        )

    if dataset == FFPP_DATASET:
        if study_label == 0:
            if manipulation != "":
                raise ValueError(
                    "FF++ real record must not contain "
                    "a manipulation label."
                )

            if target_video_id != "":
                raise ValueError(
                    "FF++ real record must not contain "
                    "target_video_id."
                )

        else:
            if (
                manipulation
                not in FFPP_MANIPULATIONS
            ):
                raise ValueError(
                    "FF++ manipulated record has "
                    "unsupported manipulation: {!r}."
                    .format(
                        manipulation
                    )
                )

            if target_video_id == "":
                raise ValueError(
                    "FF++ manipulated record requires "
                    "target_video_id."
                )


def build_manifest_index(
    rows: Iterable[
        Dict[str, Any]
    ],
) -> Dict[
    Tuple[str, str],
    Dict[str, Any],
]:
    index = {}

    count = 0

    for row_number, row in enumerate(
        rows,
        start=1,
    ):
        try:
            validate_manifest_record(
                row
            )

            key = manifest_key(
                row
            )
        except ValueError as exc:
            raise ValueError(
                "Invalid canonical manifest row #{}: {}"
                .format(
                    row_number,
                    exc,
                )
            ) from exc

        if key in index:
            raise ValueError(
                "Duplicate canonical manifest identity: {}."
                .format(
                    key
                )
            )

        index[
            key
        ] = row

        count += 1

    if count == 0:
        raise ValueError(
            "Canonical manifest contains no records."
        )

    return index


def expected_bootstrap_stratum(
    *,
    dataset: str,
    study_label: int,
    manipulation: str,
) -> str:
    if study_label == 0:
        return "real"

    if dataset == FFPP_DATASET:
        return manipulation

    if dataset == CELEB_DATASET:
        return "manipulated"

    raise ValueError(
        "Unsupported dataset: {}.".format(
            dataset
        )
    )


def expected_bootstrap_unit_id(
    *,
    dataset: str,
    source_video_id: str,
    base_video_id: str,
) -> str:
    if dataset == FFPP_DATASET:
        return source_video_id

    if dataset == CELEB_DATASET:
        return base_video_id

    raise ValueError(
        "Unsupported dataset: {}.".format(
            dataset
        )
    )


def validate_equal(
    *,
    expected: Any,
    actual: Any,
    field_name: str,
    key: Tuple[str, str, str, str],
    stage: str,
) -> None:
    if expected != actual:
        raise ValueError(
            "{} changed canonical field {} for {}: "
            "{!r} != {!r}.".format(
                stage,
                field_name,
                key,
                expected,
                actual,
            )
        )


def validate_manifest_analysis_chain(
    *,
    manifest_index: Dict[
        Tuple[str, str],
        Dict[str, Any],
    ],
    inference_records: List[
        Dict[str, Any]
    ],
    analysis_records: List[
        Dict[str, Any]
    ],
) -> None:
    inference_by_key = {
        inference_record_key(
            row
        ): row
        for row in inference_records
    }

    analysis_by_key = {
        analysis_record_key(
            row
        ): row
        for row in analysis_records
    }

    inference_keys = set(
        inference_by_key
    )

    analysis_keys = set(
        analysis_by_key
    )

    if inference_keys != analysis_keys:
        raise ValueError(
            "Inference and analysis key universes "
            "differ before manifest validation."
        )

    for key in sorted(
        inference_keys
    ):
        inference = inference_by_key[
            key
        ]

        analysis = analysis_by_key[
            key
        ]

        dataset = key[
            1
        ]

        relative_source_path = key[
            2
        ]

        canonical_key = (
            dataset,
            relative_source_path,
        )

        manifest = manifest_index.get(
            canonical_key
        )

        if manifest is None:
            raise ValueError(
                "Inference/analysis record is absent "
                "from canonical manifest: {}."
                .format(
                    canonical_key
                )
            )

        manifest_label = (
            strict_csv_binary_int(
                manifest[
                    "study_label"
                ],
                field_name="study_label",
            )
        )

        #
        # --------------------------------------------------
        # Manifest -> inference preservation
        # --------------------------------------------------
        #
        validate_equal(
            expected=(
                manifest[
                    "base_video_id"
                ]
            ),
            actual=(
                inference.get(
                    "base_video_id"
                )
            ),
            field_name="base_video_id",
            key=key,
            stage="Inference",
        )

        validate_equal(
            expected=manifest_label,
            actual=(
                inference.get(
                    "study_label"
                )
            ),
            field_name="study_label",
            key=key,
            stage="Inference",
        )

        validate_equal(
            expected=(
                manifest[
                    "source_label"
                ]
            ),
            actual=(
                inference.get(
                    "source_label"
                )
            ),
            field_name="source_label",
            key=key,
            stage="Inference",
        )

        #
        # --------------------------------------------------
        # Manifest -> analysis canonical preservation
        # --------------------------------------------------
        #
        canonical_fields = (
            "source_split",
            "role",
            "base_video_id",
            "source_group_id",
            "source_video_id",
            "target_video_id",
            "manipulation",
            "source_label",
        )

        for field_name in canonical_fields:
            validate_equal(
                expected=(
                    manifest[
                        field_name
                    ]
                ),
                actual=(
                    analysis.get(
                        field_name
                    )
                ),
                field_name=field_name,
                key=key,
                stage="Analysis",
            )

        validate_equal(
            expected=manifest_label,
            actual=(
                analysis.get(
                    "study_label"
                )
            ),
            field_name="study_label",
            key=key,
            stage="Analysis",
        )

        #
        # The canonical study role and the execution /
        # inference role are deliberately separate.
        #
        validate_equal(
            expected=(
                manifest[
                    "role"
                ]
            ),
            actual=(
                analysis.get(
                    "role"
                )
            ),
            field_name="role",
            key=key,
            stage="Analysis",
        )

        validate_equal(
            expected=(
                inference[
                    "role"
                ]
            ),
            actual=(
                analysis.get(
                    "inference_role"
                )
            ),
            field_name="inference_role",
            key=key,
            stage="Analysis",
        )

        #
        # --------------------------------------------------
        # Derived canonical identities
        # --------------------------------------------------
        #
        expected_video_uid = (
            "{}::{}".format(
                dataset,
                relative_source_path,
            )
        )

        validate_equal(
            expected=expected_video_uid,
            actual=(
                analysis.get(
                    "video_uid"
                )
            ),
            field_name="video_uid",
            key=key,
            stage="Analysis",
        )

        validate_equal(
            expected=expected_video_uid,
            actual=(
                analysis.get(
                    "pairing_uid"
                )
            ),
            field_name="pairing_uid",
            key=key,
            stage="Analysis",
        )

        bootstrap_stratum = (
            expected_bootstrap_stratum(
                dataset=dataset,
                study_label=(
                    manifest_label
                ),
                manipulation=(
                    manifest[
                        "manipulation"
                    ]
                ),
            )
        )

        validate_equal(
            expected=bootstrap_stratum,
            actual=(
                analysis.get(
                    "bootstrap_stratum"
                )
            ),
            field_name="bootstrap_stratum",
            key=key,
            stage="Analysis",
        )

        bootstrap_unit_id = (
            expected_bootstrap_unit_id(
                dataset=dataset,
                source_video_id=(
                    manifest[
                        "source_video_id"
                    ]
                ),
                base_video_id=(
                    manifest[
                        "base_video_id"
                    ]
                ),
            )
        )

        validate_equal(
            expected=bootstrap_unit_id,
            actual=(
                analysis.get(
                    "bootstrap_unit_id"
                )
            ),
            field_name="bootstrap_unit_id",
            key=key,
            stage="Analysis",
        )

        expected_cluster_uid = (
            "{}::{}::{}".format(
                dataset,
                manifest[
                    "role"
                ],
                bootstrap_unit_id,
            )
        )

        validate_equal(
            expected=expected_cluster_uid,
            actual=(
                analysis.get(
                    "bootstrap_cluster_uid"
                )
            ),
            field_name=(
                "bootstrap_cluster_uid"
            ),
            key=key,
            stage="Analysis",
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate canonical manifest schema and "
            "trace canonical dataset identity through "
            "inference and analysis artifacts."
        )
    )

    parser.add_argument(
        "--study-manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--inference-jsonl",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--analysis-jsonl",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    manifest_rows = read_manifest_csv(
        args.study_manifest
    )

    manifest_index = (
        build_manifest_index(
            manifest_rows
        )
    )

    from study.reproducibility.validate_analysis_artifacts import (
        read_jsonl,
    )

    inference_records = read_jsonl(
        args.inference_jsonl
    )

    analysis_records = read_jsonl(
        args.analysis_jsonl
    )

    inference_count = (
        validate_inference_records(
            inference_records
        )
    )

    analysis_count = (
        validate_analysis_records(
            analysis_records
        )
    )

    validate_manifest_analysis_chain(
        manifest_index=(
            manifest_index
        ),
        inference_records=(
            inference_records
        ),
        analysis_records=(
            analysis_records
        ),
    )

    print(
        "MANIFEST -> INFERENCE -> ANALYSIS VALIDATION"
    )
    print(
        "  canonical manifest records:   {}".format(
            len(
                manifest_index
            )
        )
    )
    print(
        "  inference records:            {}".format(
            inference_count
        )
    )
    print(
        "  analysis records:             {}".format(
            analysis_count
        )
    )
    print(
        "  strict manifest labels:       PASSED"
    )
    print(
        "  manifest identities unique:   PASSED"
    )
    print(
        "  dataset/split/role schema:    PASSED"
    )
    print(
        "  manifest -> inference label:  PASSED"
    )
    print(
        "  manifest -> inference source: PASSED"
    )
    print(
        "  canonical role preservation:  PASSED"
    )
    print(
        "  inference role preservation:  PASSED"
    )
    print(
        "  source linkage preservation:  PASSED"
    )
    print(
        "  video/pairing UID derivation: PASSED"
    )
    print(
        "  bootstrap identity derivation:PASSED"
    )
    print()
    print(
        "MANIFEST -> INFERENCE -> ANALYSIS VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()