from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from study.manifests.models import (
    DATASET_FFPP,
    LABEL_FAKE,
    LABEL_REAL,
    ROLE_DEVELOPMENT,
    ROLE_FIT,
    SOURCE_SPLIT_TRAIN,
    StudyVideoRecord,
)
from study.manifests.validation import (
    ManifestValidationError,
    validate_ffpp_records,
)


VIEW_AGGREGATE = "aggregate"
VIEW_UCF_METHOD = "ucf_method"

DB_DATASET_FACEFORENSICS = "FaceForensics++"

DB_DATASET_FF_DF = "FF-DF"
DB_DATASET_FF_F2F = "FF-F2F"
DB_DATASET_FF_FS = "FF-FS"
DB_DATASET_FF_NT = "FF-NT"


UCF_METHOD_DATASETS = (
    DB_DATASET_FF_DF,
    DB_DATASET_FF_F2F,
    DB_DATASET_FF_FS,
    DB_DATASET_FF_NT,
)


MANIPULATION_TO_DB_DATASET = {
    "DeepFakes": DB_DATASET_FF_DF,
    "Face2Face": DB_DATASET_FF_F2F,
    "FaceSwap": DB_DATASET_FF_FS,
    "NeuralTextures": DB_DATASET_FF_NT,
}


ROLE_TO_EXECUTION_SPLIT = {
    ROLE_FIT: "train",
    ROLE_DEVELOPMENT: "test",
}


EXPECTED_AGGREGATE_COUNTS = {
    ROLE_FIT: 2880,
    ROLE_DEVELOPMENT: 720,
}


EXPECTED_UCF_METHOD_COUNTS = {
    ROLE_FIT: {
        "real": 144,
        "fake": 576,
    },
    ROLE_DEVELOPMENT: {
        "real": 36,
        "fake": 144,
    },
}


MEMBERSHIP_FIELDNAMES = (
    "view",
    "adapter_dataset",
    "execution_split",
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


@dataclass(frozen=True)
class DeepfakeBenchMembershipRecord:
    """
    Map one canonical study video to one DeepfakeBench execution view.

    This object deliberately does not contain frame paths.

    Section 5.2 determines dataset membership and leakage-safe roles.
    Section 5.3 will materialize the clean training/development frame
    corpus and attach frame paths when executable DeepfakeBench JSON
    files are generated.
    """

    view: str
    adapter_dataset: str
    execution_split: str

    dataset: str
    source_split: str
    role: str

    base_video_id: str
    source_group_id: str
    source_video_id: str
    target_video_id: str

    study_label: int
    manipulation: str

    relative_source_path: str
    source_label: str


def _optional_text(
    value: str,
) -> str:
    if value is None:
        return ""

    return str(value)


def _membership_from_record(
    record: StudyVideoRecord,
    view: str,
    adapter_dataset: str,
) -> DeepfakeBenchMembershipRecord:
    execution_split = (
        ROLE_TO_EXECUTION_SPLIT.get(
            record.role
        )
    )

    if execution_split is None:
        raise ManifestValidationError(
            "Record with role {!r} cannot enter the "
            "training/development adapter."
            .format(
                record.role
            )
        )

    return DeepfakeBenchMembershipRecord(
        view=view,
        adapter_dataset=adapter_dataset,
        execution_split=execution_split,
        dataset=record.dataset,
        source_split=record.source_split,
        role=record.role,
        base_video_id=record.base_video_id,
        source_group_id=record.source_group_id,
        source_video_id=record.source_video_id,
        target_video_id=_optional_text(
            record.target_video_id
        ),
        study_label=record.study_label,
        manipulation=_optional_text(
            record.manipulation
        ),
        relative_source_path=(
            record.relative_source_path
        ),
        source_label=_optional_text(
            record.source_label
        ),
    )


def read_study_manifest(
    path: Path,
) -> List[StudyVideoRecord]:
    """
    Read the canonical CSV produced by build_manifests.py.
    """
    records = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            try:
                study_label = int(
                    row["study_label"]
                )
            except Exception as exc:
                raise ManifestValidationError(
                    "Invalid study_label in {} at CSV row {}."
                    .format(
                        path,
                        row_number,
                    )
                ) from exc

            target_video_id = (
                row["target_video_id"]
                or None
            )

            manipulation = (
                row["manipulation"]
                or None
            )

            source_label = (
                row["source_label"]
                or None
            )

            records.append(
                StudyVideoRecord(
                    dataset=row["dataset"],
                    source_split=(
                        row["source_split"]
                    ),
                    role=row["role"],
                    base_video_id=(
                        row["base_video_id"]
                    ),
                    source_group_id=(
                        row["source_group_id"]
                    ),
                    source_video_id=(
                        row["source_video_id"]
                    ),
                    target_video_id=(
                        target_video_id
                    ),
                    study_label=study_label,
                    manipulation=manipulation,
                    relative_source_path=(
                        row[
                            "relative_source_path"
                        ]
                    ),
                    source_label=source_label,
                )
            )

    return records


def select_ffpp_training_universe(
    records: Sequence[StudyVideoRecord],
) -> List[StudyVideoRecord]:
    """
    Select only FF++ FIT and internal-development records.

    Official FF++ validation, official FF++ test, and Celeb-DF-v2 are
    intentionally unreachable from this adapter.
    """
    selected = [
        record
        for record in records
        if (
            record.dataset == DATASET_FFPP
            and record.role
            in (
                ROLE_FIT,
                ROLE_DEVELOPMENT,
            )
        )
    ]

    selected.sort(
        key=lambda record: (
            record.role,
            record.source_group_id,
            record.study_label,
            record.manipulation or "",
            record.base_video_id,
        )
    )

    return selected


def build_aggregate_membership(
    records: Sequence[StudyVideoRecord],
) -> List[DeepfakeBenchMembershipRecord]:
    """
    Build the aggregate FaceForensics++ execution view.

    This view is intended for:
        Xception training,
        SPSL training,
        pooled development evaluation for all three detectors.

    Every canonical FIT/DEV video occurs exactly once.
    """
    membership = []

    for record in records:
        membership.append(
            _membership_from_record(
                record=record,
                view=VIEW_AGGREGATE,
                adapter_dataset=(
                    DB_DATASET_FACEFORENSICS
                ),
            )
        )

    return membership


def _assign_real_groups_to_ucf_views(
    records: Sequence[StudyVideoRecord],
    role: str,
) -> Dict[str, str]:
    """
    Assign complete FF++ source-pair groups to UCF method views.

    The assignment is deterministic round-robin over sorted source-group
    IDs. Both real videos from a source pair remain in the same adapter
    view.

    This is an execution-only partition. It does not alter the canonical
    study role or the experimental source-pair assignment.
    """
    source_group_ids = sorted(
        {
            record.source_group_id
            for record in records
            if (
                record.role == role
                and record.study_label
                == LABEL_REAL
            )
        }
    )

    if not source_group_ids:
        raise ManifestValidationError(
            "No real source groups found for role {!r}."
            .format(
                role
            )
        )

    if (
        len(source_group_ids)
        % len(UCF_METHOD_DATASETS)
        != 0
    ):
        raise ManifestValidationError(
            "Cannot evenly distribute {} real source groups "
            "for role {!r} across {} UCF method views."
            .format(
                len(source_group_ids),
                role,
                len(UCF_METHOD_DATASETS),
            )
        )

    assignment = {}

    for index, source_group_id in enumerate(
        source_group_ids
    ):
        adapter_dataset = (
            UCF_METHOD_DATASETS[
                index
                % len(
                    UCF_METHOD_DATASETS
                )
            ]
        )

        assignment[
            source_group_id
        ] = adapter_dataset

    return assignment


def build_ucf_method_membership(
    records: Sequence[StudyVideoRecord],
) -> List[DeepfakeBenchMembershipRecord]:
    """
    Build four UCF detector-facing method views.

    Fake videos are assigned to the view corresponding to their
    manipulation method.

    Real source pairs are distributed deterministically across the four
    method views so that, after pairDataset merges all train_dataset
    entries, every canonical real video occurs once rather than four
    times.

    pairDataset will later construct the method-specific labels using
    UCF's label_dict and will pair every fake training sample with a
    randomly selected real training sample.
    """
    real_group_assignment = {}

    for role in (
        ROLE_FIT,
        ROLE_DEVELOPMENT,
    ):
        role_assignment = (
            _assign_real_groups_to_ucf_views(
                records=records,
                role=role,
            )
        )

        for source_group_id, dataset_name in (
            role_assignment.items()
        ):
            real_group_assignment[
                (
                    role,
                    source_group_id,
                )
            ] = dataset_name

    membership = []

    for record in records:
        if record.study_label == LABEL_REAL:
            adapter_dataset = (
                real_group_assignment[
                    (
                        record.role,
                        record.source_group_id,
                    )
                ]
            )

        elif record.study_label == LABEL_FAKE:
            adapter_dataset = (
                MANIPULATION_TO_DB_DATASET.get(
                    record.manipulation
                )
            )

            if adapter_dataset is None:
                raise ManifestValidationError(
                    "Cannot map manipulation {!r} "
                    "to a UCF method dataset."
                    .format(
                        record.manipulation
                    )
                )

        else:
            raise ManifestValidationError(
                "Unexpected study label: {!r}."
                .format(
                    record.study_label
                )
            )

        membership.append(
            _membership_from_record(
                record=record,
                view=VIEW_UCF_METHOD,
                adapter_dataset=(
                    adapter_dataset
                ),
            )
        )

    return membership


def _canonical_identity(
    record: DeepfakeBenchMembershipRecord,
) -> Tuple[str, str, str]:
    return (
        record.role,
        record.source_group_id,
        record.relative_source_path,
    )


def validate_aggregate_membership(
    records: Sequence[
        DeepfakeBenchMembershipRecord
    ],
) -> None:
    role_counts = Counter(
        record.role
        for record in records
    )

    for role, expected_count in (
        EXPECTED_AGGREGATE_COUNTS.items()
    ):
        actual_count = role_counts.get(
            role,
            0,
        )

        if actual_count != expected_count:
            raise ManifestValidationError(
                "Unexpected aggregate count for role {!r}: "
                "{}; expected {}."
                .format(
                    role,
                    actual_count,
                    expected_count,
                )
            )

    identities = [
        _canonical_identity(
            record
        )
        for record in records
    ]

    if (
        len(identities)
        != len(set(identities))
    ):
        raise ManifestValidationError(
            "Duplicate canonical videos in aggregate "
            "DeepfakeBench membership."
        )


def validate_ucf_method_membership(
    records: Sequence[
        DeepfakeBenchMembershipRecord
    ],
) -> None:
    identities = [
        _canonical_identity(
            record
        )
        for record in records
    ]

    if (
        len(identities)
        != len(set(identities))
    ):
        raise ManifestValidationError(
            "A canonical FIT/DEV video appears in more than "
            "one UCF method view."
        )

    for role in (
        ROLE_FIT,
        ROLE_DEVELOPMENT,
    ):
        for dataset_name in (
            UCF_METHOD_DATASETS
        ):
            subset = [
                record
                for record in records
                if (
                    record.role == role
                    and record.adapter_dataset
                    == dataset_name
                )
            ]

            real_count = sum(
                1
                for record in subset
                if (
                    record.study_label
                    == LABEL_REAL
                )
            )

            fake_count = sum(
                1
                for record in subset
                if (
                    record.study_label
                    == LABEL_FAKE
                )
            )

            expected = (
                EXPECTED_UCF_METHOD_COUNTS[
                    role
                ]
            )

            if (
                real_count
                != expected["real"]
            ):
                raise ManifestValidationError(
                    "Unexpected real count for UCF view "
                    "{!r}, role {!r}: {}; expected {}."
                    .format(
                        dataset_name,
                        role,
                        real_count,
                        expected["real"],
                    )
                )

            if (
                fake_count
                != expected["fake"]
            ):
                raise ManifestValidationError(
                    "Unexpected fake count for UCF view "
                    "{!r}, role {!r}: {}; expected {}."
                    .format(
                        dataset_name,
                        role,
                        fake_count,
                        expected["fake"],
                    )
                )

            for record in subset:
                if (
                    record.study_label
                    == LABEL_FAKE
                ):
                    expected_dataset = (
                        MANIPULATION_TO_DB_DATASET[
                            record.manipulation
                        ]
                    )

                    if (
                        record.adapter_dataset
                        != expected_dataset
                    ):
                        raise ManifestValidationError(
                            "Fake record {!r} entered "
                            "the wrong UCF method view."
                            .format(
                                record.relative_source_path
                            )
                        )


def validate_no_held_out_membership(
    records: Sequence[
        DeepfakeBenchMembershipRecord
    ],
) -> None:
    for record in records:
        if record.role not in (
            ROLE_FIT,
            ROLE_DEVELOPMENT,
        ):
            raise ManifestValidationError(
                "Held-out role leaked into DeepfakeBench "
                "training adapter: {!r}."
                .format(
                    record.role
                )
            )

        if (
            record.source_split
            != SOURCE_SPLIT_TRAIN
        ):
            raise ManifestValidationError(
                "Non-training FF++ native split leaked into "
                "DeepfakeBench training adapter: "
                "role={!r}, source_split={!r}, path={!r}."
                .format(
                    record.role,
                    record.source_split,
                    record.relative_source_path,
                )
            )

        expected_execution_split = (
            ROLE_TO_EXECUTION_SPLIT[
                record.role
            ]
        )

        if (
            record.execution_split
            != expected_execution_split
        ):
            raise ManifestValidationError(
                "Adapter execution-split mismatch for {!r}: "
                "{!r}; expected {!r}."
                .format(
                    record.relative_source_path,
                    record.execution_split,
                    expected_execution_split,
                )
            )


def validate_view_equivalence(
    aggregate_records: Sequence[
        DeepfakeBenchMembershipRecord
    ],
    ucf_records: Sequence[
        DeepfakeBenchMembershipRecord
    ],
) -> None:
    """
    Require aggregate and UCF views to represent exactly the same
    canonical FIT/DEV video universe.
    """
    aggregate_identities = {
        _canonical_identity(
            record
        )
        for record in aggregate_records
    }

    ucf_identities = {
        _canonical_identity(
            record
        )
        for record in ucf_records
    }

    if (
        aggregate_identities
        != ucf_identities
    ):
        only_aggregate = sorted(
            aggregate_identities
            - ucf_identities
        )

        only_ucf = sorted(
            ucf_identities
            - aggregate_identities
        )

        raise ManifestValidationError(
            "Aggregate and UCF adapter views do not represent "
            "the same canonical FIT/DEV universe. "
            "Only aggregate: {}. Only UCF: {}."
            .format(
                only_aggregate[:20],
                only_ucf[:20],
            )
        )


def sort_membership(
    records: Sequence[
        DeepfakeBenchMembershipRecord
    ],
) -> List[
    DeepfakeBenchMembershipRecord
]:
    return sorted(
        records,
        key=lambda record: (
            record.view,
            record.execution_split,
            record.adapter_dataset,
            record.source_group_id,
            record.study_label,
            record.manipulation,
            record.base_video_id,
        ),
    )


def write_membership_csv(
    records: Sequence[
        DeepfakeBenchMembershipRecord
    ],
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
            fieldnames=(
                MEMBERSHIP_FIELDNAMES
            ),
            lineterminator="\n",
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                asdict(record)
            )


def write_json(
    data: object,
    path: Path,
) -> None:
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


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
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


def make_summary(
    records: Sequence[
        DeepfakeBenchMembershipRecord
    ],
) -> Dict[str, object]:
    counts = Counter(
        (
            record.view,
            record.execution_split,
            record.adapter_dataset,
            record.study_label,
        )
        for record in records
    )

    return {
        "total_membership_rows": len(
            records
        ),
        "counts": {
            "{}|{}|{}|{}".format(
                view,
                execution_split,
                dataset_name,
                study_label,
            ): count
            for (
                view,
                execution_split,
                dataset_name,
                study_label,
            ), count in sorted(
                counts.items()
            )
        },
    }


def build_adapter_membership(
    study_manifest_path: Path,
    output_dir: Path,
) -> None:
    records = read_study_manifest(
        study_manifest_path
    )

    ffpp_records = [
        record
        for record in records
        if record.dataset == DATASET_FFPP
    ]

    validate_ffpp_records(
        ffpp_records
    )

    training_universe = (
        select_ffpp_training_universe(
            records
        )
    )

    aggregate = (
        build_aggregate_membership(
            training_universe
        )
    )

    ucf_method = (
        build_ucf_method_membership(
            training_universe
        )
    )

    validate_aggregate_membership(
        aggregate
    )

    validate_ucf_method_membership(
        ucf_method
    )

    validate_no_held_out_membership(
        aggregate
    )

    validate_no_held_out_membership(
        ucf_method
    )

    validate_view_equivalence(
        aggregate_records=aggregate,
        ucf_records=ucf_method,
    )

    membership = sort_membership(
        list(aggregate)
        + list(ucf_method)
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    membership_path = (
        output_dir
        / "deepfakebench_membership.csv"
    )

    summary_path = (
        output_dir
        / "deepfakebench_membership_summary.json"
    )

    metadata_path = (
        output_dir
        / "deepfakebench_membership_metadata.json"
    )

    write_membership_csv(
        records=membership,
        path=membership_path,
    )

    write_json(
        make_summary(
            membership
        ),
        summary_path,
    )

    metadata = {
        "schema_version": 1,
        "source_manifest": {
            "filename": (
                study_manifest_path.name
            ),
            "sha256": sha256_file(
                study_manifest_path
            ),
        },
        "execution_contract": {
            "fit_role": "train",
            "development_role": "test",
            "aggregate_dataset": (
                DB_DATASET_FACEFORENSICS
            ),
            "ucf_method_datasets": list(
                UCF_METHOD_DATASETS
            ),
            "contains_official_validation": False,
            "contains_official_test": False,
            "contains_external_evaluation": False,
            "contains_frame_paths": False,
        },
        "outputs": {
            membership_path.name: {
                "sha256": sha256_file(
                    membership_path
                ),
            },
            summary_path.name: {
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
        "DEEPFAKEBENCH MEMBERSHIP BUILT"
    )
    print()

    print(
        "Canonical FIT/DEV videos:  {}"
        .format(
            len(training_universe)
        )
    )

    print(
        "Aggregate view rows:       {}"
        .format(
            len(aggregate)
        )
    )

    print(
        "UCF method-view rows:      {}"
        .format(
            len(ucf_method)
        )
    )

    print(
        "Total adapter rows:        {}"
        .format(
            len(membership)
        )
    )

    print()
    print(
        "Membership:               {}"
        .format(
            membership_path
        )
    )

    print(
        "Summary:                  {}"
        .format(
            summary_path
        )
    )

    print(
        "Metadata:                 {}"
        .format(
            metadata_path
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the leakage-safe DeepfakeBench "
            "FIT/development membership adapter."
        )
    )

    parser.add_argument(
        "--study-manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    build_adapter_membership(
        study_manifest_path=(
            args.study_manifest
        ),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()