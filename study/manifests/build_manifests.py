from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

from study.manifests.celeb_df import (
    load_celeb_df_v2_test_records,
)
from study.manifests.ffpp import (
    build_ffpp_study_records,
    load_official_pairs,
)
from study.manifests.models import (
    DATASET_CELEB_DF_V2,
    DATASET_FFPP,
    LABEL_FAKE,
    LABEL_REAL,
    SourcePair,
    StudyVideoRecord,
    record_to_dict,
)
from study.manifests.validation import (
    ManifestValidationError,
    validate_celeb_df_v2_records,
    validate_ffpp_records,
    validate_source_files,
)


DEFAULT_SPLIT_SEED = 1024

EXPECTED_CELEB_TEST_RECORDS = 518
EXPECTED_CELEB_REAL_RECORDS = 178
EXPECTED_CELEB_FAKE_RECORDS = 340


MANIFEST_FIELDNAMES = (
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


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def write_json(
    data: object,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

        file.write("\n")


def write_manifest_csv(
    records: Sequence[StudyVideoRecord],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=MANIFEST_FIELDNAMES,
            lineterminator="\n",
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                record_to_dict(
                    record
                )
            )


def serialize_pairs(
    pairs: Sequence[SourcePair],
) -> List[List[str]]:
    return [
        [
            first,
            second,
        ]
        for first, second in pairs
    ]


def validate_celeb_expected_composition(
    records: Sequence[StudyVideoRecord],
) -> None:
    if (
        len(records)
        != EXPECTED_CELEB_TEST_RECORDS
    ):
        raise ManifestValidationError(
            "Unexpected Celeb-DF-v2 test record count: {}; "
            "expected {}."
            .format(
                len(records),
                EXPECTED_CELEB_TEST_RECORDS,
            )
        )

    label_counts = Counter(
        record.study_label
        for record in records
    )

    actual_real = label_counts.get(
        LABEL_REAL,
        0,
    )

    actual_fake = label_counts.get(
        LABEL_FAKE,
        0,
    )

    if (
        actual_real
        != EXPECTED_CELEB_REAL_RECORDS
    ):
        raise ManifestValidationError(
            "Unexpected Celeb-DF-v2 real record count: {}; "
            "expected {}."
            .format(
                actual_real,
                EXPECTED_CELEB_REAL_RECORDS,
            )
        )

    if (
        actual_fake
        != EXPECTED_CELEB_FAKE_RECORDS
    ):
        raise ManifestValidationError(
            "Unexpected Celeb-DF-v2 fake record count: {}; "
            "expected {}."
            .format(
                actual_fake,
                EXPECTED_CELEB_FAKE_RECORDS,
            )
        )


def sort_records(
    records: Sequence[StudyVideoRecord],
) -> List[StudyVideoRecord]:
    return sorted(
        records,
        key=lambda record: (
            record.dataset,
            record.role,
            record.source_group_id,
            record.study_label,
            record.manipulation or "",
            record.base_video_id,
            record.relative_source_path,
        ),
    )


def make_summary(
    records: Sequence[StudyVideoRecord],
) -> Dict[str, object]:
    dataset_counts = Counter(
        record.dataset
        for record in records
    )

    dataset_role_counts = Counter(
        (
            record.dataset,
            record.role,
        )
        for record in records
    )

    dataset_role_label_counts = Counter(
        (
            record.dataset,
            record.role,
            record.study_label,
        )
        for record in records
    )

    ffpp_manipulation_counts = Counter(
        record.manipulation
        for record in records
        if (
            record.dataset == DATASET_FFPP
            and record.study_label == LABEL_FAKE
        )
    )

    return {
        "total_records": len(records),
        "dataset_counts": {
            dataset: count
            for dataset, count in sorted(
                dataset_counts.items()
            )
        },
        "dataset_role_counts": {
            "{}|{}".format(
                dataset,
                role,
            ): count
            for (
                dataset,
                role,
            ), count in sorted(
                dataset_role_counts.items()
            )
        },
        "dataset_role_label_counts": {
            "{}|{}|{}".format(
                dataset,
                role,
                label,
            ): count
            for (
                dataset,
                role,
                label,
            ), count in sorted(
                dataset_role_label_counts.items()
            )
        },
        "ffpp_fake_manipulation_counts": {
            manipulation: count
            for manipulation, count in sorted(
                ffpp_manipulation_counts.items()
            )
        },
    }


def build_artifacts(
    ffpp_root: Path,
    ffpp_splits_dir: Path,
    celeb_root: Path,
    celeb_test_list: Path,
    output_dir: Path,
    split_seed: int,
) -> None:
    train_path = (
        ffpp_splits_dir
        / "train.json"
    )

    val_path = (
        ffpp_splits_dir
        / "val.json"
    )

    test_path = (
        ffpp_splits_dir
        / "test.json"
    )

    for path in (
        train_path,
        val_path,
        test_path,
        celeb_test_list,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                "Required manifest source does not exist: {}"
                .format(path)
            )

    if not ffpp_root.is_dir():
        raise FileNotFoundError(
            "FF++ root does not exist: {}"
            .format(ffpp_root)
        )

    if not celeb_root.is_dir():
        raise FileNotFoundError(
            "Celeb-DF-v2 root does not exist: {}"
            .format(celeb_root)
        )

    train_pairs = load_official_pairs(
        train_path
    )

    val_pairs = load_official_pairs(
        val_path
    )

    test_pairs = load_official_pairs(
        test_path
    )

    (
        ffpp_records,
        fit_pairs,
        development_pairs,
    ) = build_ffpp_study_records(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        split_seed=split_seed,
    )

    validate_ffpp_records(
        ffpp_records
    )

    celeb_records = (
        load_celeb_df_v2_test_records(
            celeb_test_list
        )
    )

    validate_celeb_df_v2_records(
        celeb_records
    )

    validate_celeb_expected_composition(
        celeb_records
    )

    records = sort_records(
        list(ffpp_records)
        + list(celeb_records)
    )

    validate_source_files(
        records=records,
        dataset_roots={
            DATASET_FFPP: ffpp_root,
            DATASET_CELEB_DF_V2: celeb_root,
        },
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        output_dir
        / "study_manifest.csv"
    )

    assignment_path = (
        output_dir
        / "ffpp_train_pair_assignment.json"
    )

    summary_path = (
        output_dir
        / "manifest_summary.json"
    )

    metadata_path = (
        output_dir
        / "manifest_metadata.json"
    )

    write_manifest_csv(
        records=records,
        path=manifest_path,
    )

    assignment = {
        "split_seed": split_seed,
        "official_train_pairs": len(
            train_pairs
        ),
        "fit_pairs": serialize_pairs(
            fit_pairs
        ),
        "development_pairs": serialize_pairs(
            development_pairs
        ),
    }

    write_json(
        assignment,
        assignment_path,
    )

    summary = make_summary(
        records
    )

    write_json(
        summary,
        summary_path,
    )

    metadata = {
        "schema_version": 1,
        "split_seed": split_seed,
        "inputs": {
            "ffpp_train_split": {
                "filename": train_path.name,
                "sha256": sha256_file(
                    train_path
                ),
            },
            "ffpp_validation_split": {
                "filename": val_path.name,
                "sha256": sha256_file(
                    val_path
                ),
            },
            "ffpp_test_split": {
                "filename": test_path.name,
                "sha256": sha256_file(
                    test_path
                ),
            },
            "celeb_df_v2_test_list": {
                "filename": (
                    celeb_test_list.name
                ),
                "sha256": sha256_file(
                    celeb_test_list
                ),
            },
        },
        "outputs": {
            "study_manifest.csv": {
                "sha256": sha256_file(
                    manifest_path
                ),
            },
            "ffpp_train_pair_assignment.json": {
                "sha256": sha256_file(
                    assignment_path
                ),
            },
            "manifest_summary.json": {
                "sha256": sha256_file(
                    summary_path
                ),
            },
        },
    }

    write_json(
        metadata,
        metadata_path,
    )

    print(
        "STUDY MANIFEST ARTIFACTS BUILT"
    )
    print()

    print(
        "Split seed:              {}"
        .format(
            split_seed
        )
    )

    print(
        "FF++ records:            {}"
        .format(
            len(ffpp_records)
        )
    )

    print(
        "Celeb-DF-v2 records:     {}"
        .format(
            len(celeb_records)
        )
    )

    print(
        "Total records:           {}"
        .format(
            len(records)
        )
    )

    print(
        "FIT pairs:               {}"
        .format(
            len(fit_pairs)
        )
    )

    print(
        "Development pairs:       {}"
        .format(
            len(development_pairs)
        )
    )

    print()
    print(
        "Manifest:                {}"
        .format(
            manifest_path
        )
    )

    print(
        "Pair assignment:         {}"
        .format(
            assignment_path
        )
    )

    print(
        "Summary:                 {}"
        .format(
            summary_path
        )
    )

    print(
        "Metadata:                {}"
        .format(
            metadata_path
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build and validate the canonical "
            "study-level dataset manifests."
        )
    )

    parser.add_argument(
        "--ffpp-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--ffpp-splits-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--celeb-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--celeb-test-list",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
    )

    args = parser.parse_args()

    build_artifacts(
        ffpp_root=args.ffpp_root,
        ffpp_splits_dir=(
            args.ffpp_splits_dir
        ),
        celeb_root=args.celeb_root,
        celeb_test_list=(
            args.celeb_test_list
        ),
        output_dir=args.output_dir,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()