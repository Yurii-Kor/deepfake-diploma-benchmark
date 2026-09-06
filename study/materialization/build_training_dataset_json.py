"""
Build executable DeepfakeBench training JSON files from the frozen study
membership adapter and the completed frame-materialization audit.

Source-of-truth separation
--------------------------
study/manifests/artifacts/deepfakebench_membership.csv
    Defines which canonical video belongs to which DeepfakeBench execution
    view and split.

study/materialization/artifacts/frame_materialization.csv
    Defines which of the 32 nominal temporal targets produced valid aligned
    detector inputs.

Only rows with materialization_status == "ok" are exposed to detector
training/development execution. Failed target positions remain preserved in
the materialization audit and are never replaced here.

Generated execution views
-------------------------
FaceForensics++.json
    Aggregate FIT / DEV view used by Xception and SPSL.

FF-DF.json
FF-F2F.json
FF-FS.json
FF-NT.json
    Method-specific UCF execution views.

Canonical FIT is mapped to DeepfakeBench "train".
Canonical development is mapped to DeepfakeBench "test".

The generated JSON files contain only references to already materialized
256x256 face crops. No frame extraction, face detection, replacement,
padding, or resampling is performed by this script.
"""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MEMBERSHIP = (
    PROJECT_ROOT
    / "study"
    / "manifests"
    / "artifacts"
    / "deepfakebench_membership.csv"
)

DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "study"
    / "materialization"
    / "artifacts"
    / "frame_materialization.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "study"
    / "training"
    / "materialized"
    / "dataset_json"
)

DEFAULT_RGB_ROOT = (
    PROJECT_ROOT
    / "study"
    / "training"
    / "materialized"
    / "rgb"
)

DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "study"
    / "materialization"
    / "artifacts"
    / "training_dataset_json_summary.json"
)


EXPECTED_AUDIT_ROWS = 115200
EXPECTED_SUCCESSFUL_TARGETS = 115099
EXPECTED_FAILED_TARGETS = 101

EXPECTED_MEMBERSHIP_ROWS = 7200
EXPECTED_AGGREGATE_VIDEOS = 3600
EXPECTED_UCF_UNION_VIDEOS = 3600

EXPECTED_ADAPTERS = (
    "FaceForensics++",
    "FF-DF",
    "FF-F2F",
    "FF-FS",
    "FF-NT",
)

EXPECTED_LABELS = {
    "FaceForensics++": (
        "FF-real",
        "FF-DF",
        "FF-F2F",
        "FF-FS",
        "FF-NT",
    ),
    "FF-DF": (
        "FF-real",
        "FF-DF",
    ),
    "FF-F2F": (
        "FF-real",
        "FF-F2F",
    ),
    "FF-FS": (
        "FF-real",
        "FF-FS",
    ),
    "FF-NT": (
        "FF-real",
        "FF-NT",
    ),
}

EXPECTED_ROLE_TO_SPLIT = {
    "fit": "train",
    "development": "test",
}

MANIPULATION_TO_DFB_LABEL = {
    "DeepFakes": "FF-DF",
    "Face2Face": "FF-F2F",
    "FaceSwap": "FF-FS",
    "NeuralTextures": "FF-NT",
}

EXPECTED_AGGREGATE_FRAME_COUNTS = {
    ("fit", "FF-real"): 18426,
    ("fit", "FF-DF"): 18398,
    ("fit", "FF-F2F"): 18423,
    ("fit", "FF-FS"): 18423,
    ("fit", "FF-NT"): 18407,
    ("development", "FF-real"): 4608,
    ("development", "FF-DF"): 4593,
    ("development", "FF-F2F"): 4608,
    ("development", "FF-FS"): 4606,
    ("development", "FF-NT"): 4607,
}


def sha256_file(path):
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


def read_csv(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(
            csv.DictReader(
                file
            )
        )


def require_columns(
    rows,
    required,
    source_name,
):
    if not rows:
        raise ValueError(
            "{} contains no rows".format(
                source_name
            )
        )

    missing = sorted(
        set(required)
        - set(rows[0])
    )

    if missing:
        raise ValueError(
            "{} is missing required columns: {}".format(
                source_name,
                missing,
            )
        )


def normalize_relative_path(
    value,
):
    normalized = str(
        value
    ).replace(
        "\\",
        "/",
    )

    while normalized.startswith(
        "./"
    ):
        normalized = normalized[
            2:
        ]

    return normalized


def derive_dfb_label(
    study_label,
    manipulation,
):
    """
    Derive the DeepfakeBench execution label from canonical study semantics.

    Canonical membership is authoritative:

        study_label == 0
            -> FF-real

        study_label == 1 + DeepFakes
            -> FF-DF

        study_label == 1 + Face2Face
            -> FF-F2F

        study_label == 1 + FaceSwap
            -> FF-FS

        study_label == 1 + NeuralTextures
            -> FF-NT

    The membership CSV's source_label field is deliberately not used here.
    In the frozen membership artifact it is empty; study_label and
    manipulation are the canonical fields that determine execution labels.
    """

    normalized_study_label = str(
        study_label
    ).strip()

    normalized_manipulation = str(
        manipulation
    ).strip()

    if normalized_study_label == "0":
        if normalized_manipulation:
            raise ValueError(
                "real membership row must not define manipulation; "
                "got {!r}".format(
                    normalized_manipulation
                )
            )

        return "FF-real"

    if normalized_study_label == "1":
        if (
            normalized_manipulation
            not in MANIPULATION_TO_DFB_LABEL
        ):
            raise ValueError(
                "fake membership row has unsupported manipulation: "
                "{!r}".format(
                    normalized_manipulation
                )
            )

        return MANIPULATION_TO_DFB_LABEL[
            normalized_manipulation
        ]

    raise ValueError(
        "study_label must be 0 or 1, got {!r}".format(
            study_label
        )
    )


def video_name_from_relative_path(
    relative_source_path,
):
    normalized = normalize_relative_path(
        relative_source_path
    )

    return Path(
        normalized
    ).stem


def empty_dataset_structure(
    adapter_dataset,
):
    labels = EXPECTED_LABELS[
        adapter_dataset
    ]

    return {
        adapter_dataset: {
            label: {
                "train": {
                    "raw": {},
                },
                "val": {
                    "raw": {},
                },
                "test": {
                    "raw": {},
                },
            }
            for label in labels
        }
    }


def write_json_atomic(
    path,
    data,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write(
            "\n"
        )

    os.replace(
        str(
            temporary
        ),
        str(
            path
        ),
    )


def build_successful_frame_index(
    audit_rows,
    rgb_root,
):
    successful = defaultdict(
        list
    )

    successful_output_paths = set()

    status_counts = Counter(
        row[
            "materialization_status"
        ]
        for row in audit_rows
    )

    successful_count = (
        status_counts.get(
            "ok",
            0,
        )
    )

    failed_count = (
        len(
            audit_rows
        )
        - successful_count
    )

    if (
        len(audit_rows)
        != EXPECTED_AUDIT_ROWS
    ):
        raise ValueError(
            "expected {} materialization audit rows, got {}".format(
                EXPECTED_AUDIT_ROWS,
                len(
                    audit_rows
                ),
            )
        )

    if (
        successful_count
        != EXPECTED_SUCCESSFUL_TARGETS
    ):
        raise ValueError(
            "expected {} successful materialized targets, got {}".format(
                EXPECTED_SUCCESSFUL_TARGETS,
                successful_count,
            )
        )

    if (
        failed_count
        != EXPECTED_FAILED_TARGETS
    ):
        raise ValueError(
            "expected {} failed materialized targets, got {}".format(
                EXPECTED_FAILED_TARGETS,
                failed_count,
            )
        )

    for row in audit_rows:
        if (
            row[
                "materialization_status"
            ]
            != "ok"
        ):
            continue

        role = str(
            row["role"]
        ).strip()

        relative_source_path = (
            normalize_relative_path(
                row[
                    "relative_source_path"
                ]
            )
        )

        output_relative_path = (
            normalize_relative_path(
                row[
                    "output_relative_path"
                ]
            )
        )

        temporal_position = int(
            row[
                "temporal_position"
            ]
        )

        if (
            output_relative_path
            in successful_output_paths
        ):
            raise ValueError(
                "duplicate successful output path in materialization "
                "audit: {}".format(
                    output_relative_path
                )
            )

        physical_path = (
            rgb_root
            / output_relative_path
        )

        if not physical_path.is_file():
            raise FileNotFoundError(
                "materialized output referenced by audit does not "
                "exist: {}".format(
                    physical_path
                )
            )

        successful_output_paths.add(
            output_relative_path
        )

        successful[
            (
                role,
                relative_source_path,
            )
        ].append(
            (
                temporal_position,
                output_relative_path,
            )
        )

    if (
        len(
            successful_output_paths
        )
        != EXPECTED_SUCCESSFUL_TARGETS
    ):
        raise ValueError(
            "successful materialized output-path set has size {}, "
            "expected {}".format(
                len(
                    successful_output_paths
                ),
                EXPECTED_SUCCESSFUL_TARGETS,
            )
        )

    for key, values in successful.items():
        values.sort(
            key=lambda item: item[0]
        )

        temporal_positions = [
            item[0]
            for item in values
        ]

        if (
            len(
                temporal_positions
            )
            != len(
                set(
                    temporal_positions
                )
            )
        ):
            raise ValueError(
                "duplicate successful temporal positions for {}".format(
                    key
                )
            )

        if len(values) > 32:
            raise ValueError(
                "video {} has {} successful targets; "
                "maximum is 32".format(
                    key,
                    len(
                        values
                    ),
                )
            )

    return (
        successful,
        successful_output_paths,
        status_counts,
    )


def validate_membership(
    membership_rows,
):
    if (
        len(
            membership_rows
        )
        != EXPECTED_MEMBERSHIP_ROWS
    ):
        raise ValueError(
            "expected {} membership rows, got {}".format(
                EXPECTED_MEMBERSHIP_ROWS,
                len(
                    membership_rows
                ),
            )
        )

    seen = set()
    aggregate_keys = set()
    ucf_keys = set()

    adapter_counts = Counter()
    role_counts = Counter()

    for row in membership_rows:
        adapter_dataset = str(
            row[
                "adapter_dataset"
            ]
        ).strip()

        role = str(
            row["role"]
        ).strip()

        execution_split = str(
            row[
                "execution_split"
            ]
        ).strip()

        dfb_label = derive_dfb_label(
            study_label=row[
                "study_label"
            ],
            manipulation=row[
                "manipulation"
            ],
        )

        relative_source_path = (
            normalize_relative_path(
                row[
                    "relative_source_path"
                ]
            )
        )

        if (
            adapter_dataset
            not in EXPECTED_ADAPTERS
        ):
            raise ValueError(
                "unexpected adapter_dataset: {!r}".format(
                    adapter_dataset
                )
            )

        if (
            role
            not in EXPECTED_ROLE_TO_SPLIT
        ):
            raise ValueError(
                "unexpected study role: {!r}".format(
                    role
                )
            )

        expected_split = (
            EXPECTED_ROLE_TO_SPLIT[
                role
            ]
        )

        if (
            execution_split
            != expected_split
        ):
            raise ValueError(
                "role {!r} must map to execution split {!r}, "
                "got {!r}".format(
                    role,
                    expected_split,
                    execution_split,
                )
            )

        if (
            dfb_label
            not in EXPECTED_LABELS[
                adapter_dataset
            ]
        ):
            raise ValueError(
                "derived DeepfakeBench label {!r} is invalid for "
                "adapter {!r}".format(
                    dfb_label,
                    adapter_dataset,
                )
            )

        row_key = (
            row[
                "view"
            ],
            adapter_dataset,
            role,
            relative_source_path,
        )

        if row_key in seen:
            raise ValueError(
                "duplicate membership row: {}".format(
                    row_key
                )
            )

        seen.add(
            row_key
        )

        canonical_key = (
            role,
            relative_source_path,
        )

        if (
            adapter_dataset
            == "FaceForensics++"
        ):
            if (
                canonical_key
                in aggregate_keys
            ):
                raise ValueError(
                    "aggregate adapter duplicates canonical video "
                    "{}".format(
                        canonical_key
                    )
                )

            aggregate_keys.add(
                canonical_key
            )

        else:
            if (
                canonical_key
                in ucf_keys
            ):
                raise ValueError(
                    "UCF execution views duplicate canonical video "
                    "{}".format(
                        canonical_key
                    )
                )

            ucf_keys.add(
                canonical_key
            )

        adapter_counts[
            adapter_dataset
        ] += 1

        role_counts[
            (
                adapter_dataset,
                role,
            )
        ] += 1

    if (
        len(
            aggregate_keys
        )
        != EXPECTED_AGGREGATE_VIDEOS
    ):
        raise ValueError(
            "aggregate adapter contains {} canonical videos; "
            "expected {}".format(
                len(
                    aggregate_keys
                ),
                EXPECTED_AGGREGATE_VIDEOS,
            )
        )

    if (
        len(
            ucf_keys
        )
        != EXPECTED_UCF_UNION_VIDEOS
    ):
        raise ValueError(
            "UCF adapter union contains {} canonical videos; "
            "expected {}".format(
                len(
                    ucf_keys
                ),
                EXPECTED_UCF_UNION_VIDEOS,
            )
        )

    if (
        aggregate_keys
        != ucf_keys
    ):
        only_aggregate = sorted(
            aggregate_keys
            - ucf_keys
        )[:10]

        only_ucf = sorted(
            ucf_keys
            - aggregate_keys
        )[:10]

        raise ValueError(
            "aggregate and UCF adapter universes differ; "
            "only aggregate sample={}, only UCF sample={}".format(
                only_aggregate,
                only_ucf,
            )
        )

    return {
        "aggregate_keys": aggregate_keys,
        "ucf_keys": ucf_keys,
        "adapter_counts": adapter_counts,
        "role_counts": role_counts,
    }


def build_datasets(
    membership_rows,
    successful_frames,
):
    datasets = {
        adapter: empty_dataset_structure(
            adapter
        )
        for adapter in EXPECTED_ADAPTERS
    }

    frame_counts = Counter()
    video_counts = Counter()

    aggregate_frame_paths = []
    ucf_frame_paths = []

    for row in membership_rows:
        adapter_dataset = str(
            row[
                "adapter_dataset"
            ]
        ).strip()

        role = str(
            row["role"]
        ).strip()

        execution_split = str(
            row[
                "execution_split"
            ]
        ).strip()

        dfb_label = derive_dfb_label(
            study_label=row[
                "study_label"
            ],
            manipulation=row[
                "manipulation"
            ],
        )

        relative_source_path = (
            normalize_relative_path(
                row[
                    "relative_source_path"
                ]
            )
        )

        key = (
            role,
            relative_source_path,
        )

        materialized = (
            successful_frames.get(
                key,
                [],
            )
        )

        if not materialized:
            raise ValueError(
                "membership video has no successful materialized "
                "frames: {}".format(
                    key
                )
            )

        frame_paths = [
            output_relative_path
            for (
                _,
                output_relative_path,
            ) in materialized
        ]

        if len(frame_paths) > 32:
            raise ValueError(
                "{} has more than 32 executable frames".format(
                    key
                )
            )

        video_name = (
            video_name_from_relative_path(
                relative_source_path
            )
        )

        target = (
            datasets[
                adapter_dataset
            ][
                adapter_dataset
            ][
                dfb_label
            ][
                execution_split
            ][
                "raw"
            ]
        )

        if (
            video_name
            in target
        ):
            raise ValueError(
                "duplicate video name {!r} in adapter={}, "
                "label={}, split={}".format(
                    video_name,
                    adapter_dataset,
                    dfb_label,
                    execution_split,
                )
            )

        target[
            video_name
        ] = {
            "label": dfb_label,
            "frames": frame_paths,
        }

        video_counts[
            (
                adapter_dataset,
                role,
                dfb_label,
            )
        ] += 1

        frame_counts[
            (
                adapter_dataset,
                role,
                dfb_label,
            )
        ] += len(
            frame_paths
        )

        if (
            adapter_dataset
            == "FaceForensics++"
        ):
            aggregate_frame_paths.extend(
                frame_paths
            )
        else:
            ucf_frame_paths.extend(
                frame_paths
            )

    return {
        "datasets": datasets,
        "frame_counts": frame_counts,
        "video_counts": video_counts,
        "aggregate_frame_paths": aggregate_frame_paths,
        "ucf_frame_paths": ucf_frame_paths,
    }


def validate_generated_views(
    built,
    successful_output_paths,
):
    aggregate_frame_paths = built[
        "aggregate_frame_paths"
    ]

    ucf_frame_paths = built[
        "ucf_frame_paths"
    ]

    if (
        len(
            aggregate_frame_paths
        )
        != EXPECTED_SUCCESSFUL_TARGETS
    ):
        raise ValueError(
            "aggregate execution view references {} frames; "
            "expected {}".format(
                len(
                    aggregate_frame_paths
                ),
                EXPECTED_SUCCESSFUL_TARGETS,
            )
        )

    if (
        len(
            set(
                aggregate_frame_paths
            )
        )
        != EXPECTED_SUCCESSFUL_TARGETS
    ):
        raise ValueError(
            "aggregate execution view contains duplicate "
            "frame references"
        )

    if (
        set(
            aggregate_frame_paths
        )
        != successful_output_paths
    ):
        raise ValueError(
            "aggregate execution view does not exactly equal the "
            "successful materialization output set"
        )

    if (
        len(
            ucf_frame_paths
        )
        != EXPECTED_SUCCESSFUL_TARGETS
    ):
        raise ValueError(
            "UCF execution-view union references {} frames; "
            "expected {}".format(
                len(
                    ucf_frame_paths
                ),
                EXPECTED_SUCCESSFUL_TARGETS,
            )
        )

    if (
        len(
            set(
                ucf_frame_paths
            )
        )
        != EXPECTED_SUCCESSFUL_TARGETS
    ):
        raise ValueError(
            "UCF execution-view union contains duplicate "
            "frame references"
        )

    if (
        set(
            ucf_frame_paths
        )
        != successful_output_paths
    ):
        raise ValueError(
            "UCF execution-view union does not exactly equal the "
            "successful materialization output set"
        )

    frame_counts = built[
        "frame_counts"
    ]

    for (
        role,
        dfb_label,
    ), expected in (
        EXPECTED_AGGREGATE_FRAME_COUNTS.items()
    ):
        actual = frame_counts.get(
            (
                "FaceForensics++",
                role,
                dfb_label,
            ),
            0,
        )

        if actual != expected:
            raise ValueError(
                "aggregate frame count mismatch for role={} "
                "label={}: expected {}, got {}".format(
                    role,
                    dfb_label,
                    expected,
                    actual,
                )
            )


def counter_to_rows(
    counter,
):
    rows = []

    for key in sorted(
        counter
    ):
        normalized_key = key

        if not isinstance(
            normalized_key,
            tuple,
        ):
            normalized_key = (
                normalized_key,
            )

        rows.append(
            {
                "key": list(
                    normalized_key
                ),
                "count": counter[
                    key
                ],
            }
        )

    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build study-controlled DeepfakeBench dataset JSON files "
            "from frozen membership and successful materialized frames."
        )
    )

    parser.add_argument(
        "--membership",
        type=Path,
        default=DEFAULT_MEMBERSHIP,
    )

    parser.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_AUDIT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--rgb-root",
        type=Path,
        default=DEFAULT_RGB_ROOT,
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    membership_path = (
        args.membership.resolve()
    )

    audit_path = (
        args.audit.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    rgb_root = (
        args.rgb_root.resolve()
    )

    summary_path = (
        args.summary.resolve()
    )

    for path, name in (
        (
            membership_path,
            "membership",
        ),
        (
            audit_path,
            "materialization audit",
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                "{} file does not exist: {}".format(
                    name,
                    path,
                )
            )

    if not rgb_root.is_dir():
        raise FileNotFoundError(
            "materialized RGB root does not exist: {}".format(
                rgb_root
            )
        )

    membership_rows = read_csv(
        membership_path
    )

    audit_rows = read_csv(
        audit_path
    )

    require_columns(
        membership_rows,
        {
            "view",
            "adapter_dataset",
            "execution_split",
            "role",
            "study_label",
            "manipulation",
            "relative_source_path",
        },
        "membership",
    )

    require_columns(
        audit_rows,
        {
            "role",
            "relative_source_path",
            "temporal_position",
            "output_relative_path",
            "materialization_status",
        },
        "materialization audit",
    )

    membership_validation = (
        validate_membership(
            membership_rows
        )
    )

    (
        successful_frames,
        successful_output_paths,
        status_counts,
    ) = build_successful_frame_index(
        audit_rows=audit_rows,
        rgb_root=rgb_root,
    )

    aggregate_keys = (
        membership_validation[
            "aggregate_keys"
        ]
    )

    materialized_video_keys = set(
        successful_frames
    )

    if (
        materialized_video_keys
        != aggregate_keys
    ):
        missing = sorted(
            aggregate_keys
            - materialized_video_keys
        )[:10]

        unexpected = sorted(
            materialized_video_keys
            - aggregate_keys
        )[:10]

        raise ValueError(
            "successful materialization video universe does not "
            "equal aggregate membership; missing sample={}, "
            "unexpected sample={}".format(
                missing,
                unexpected,
            )
        )

    built = build_datasets(
        membership_rows=membership_rows,
        successful_frames=successful_frames,
    )

    validate_generated_views(
        built=built,
        successful_output_paths=(
            successful_output_paths
        ),
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_hashes = {}

    for adapter_dataset in EXPECTED_ADAPTERS:
        output_path = (
            output_dir
            / "{}.json".format(
                adapter_dataset
            )
        )

        write_json_atomic(
            output_path,
            built[
                "datasets"
            ][
                adapter_dataset
            ],
        )

        output_hashes[
            adapter_dataset
        ] = sha256_file(
            output_path
        )

    summary = {
        "status": "passed",
        "membership_path": str(
            membership_path
        ),
        "membership_sha256": sha256_file(
            membership_path
        ),
        "membership_rows": len(
            membership_rows
        ),
        "materialization_audit_path": str(
            audit_path
        ),
        "materialization_audit_sha256": sha256_file(
            audit_path
        ),
        "materialization_audit_rows": len(
            audit_rows
        ),
        "materialization_status_counts": dict(
            sorted(
                status_counts.items()
            )
        ),
        "successful_materialized_frames": len(
            successful_output_paths
        ),
        "aggregate_video_count": len(
            membership_validation[
                "aggregate_keys"
            ]
        ),
        "ucf_union_video_count": len(
            membership_validation[
                "ucf_keys"
            ]
        ),
        "adapter_membership_counts": dict(
            sorted(
                membership_validation[
                    "adapter_counts"
                ].items()
            )
        ),
        "video_counts": counter_to_rows(
            built[
                "video_counts"
            ]
        ),
        "frame_counts": counter_to_rows(
            built[
                "frame_counts"
            ]
        ),
        "output_dir": str(
            output_dir
        ),
        "output_sha256": dict(
            sorted(
                output_hashes.items()
            )
        ),
    }

    write_json_atomic(
        summary_path,
        summary,
    )

    print(
        "TRAINING DATASET JSON BUILD"
    )

    print(
        "  membership rows:           {}".format(
            len(
                membership_rows
            )
        )
    )

    print(
        "  audit rows:                {}".format(
            len(
                audit_rows
            )
        )
    )

    print(
        "  successful frame inputs:   {}".format(
            len(
                successful_output_paths
            )
        )
    )

    print(
        "  failed target positions:   {}".format(
            EXPECTED_FAILED_TARGETS
        )
    )

    print(
        "  aggregate videos:          {}".format(
            len(
                membership_validation[
                    "aggregate_keys"
                ]
            )
        )
    )

    print(
        "  UCF union videos:          {}".format(
            len(
                membership_validation[
                    "ucf_keys"
                ]
            )
        )
    )

    print()
    print(
        "AGGREGATE FRAME COUNTS"
    )

    for role in (
        "fit",
        "development",
    ):
        for dfb_label in EXPECTED_LABELS[
            "FaceForensics++"
        ]:
            count = built[
                "frame_counts"
            ].get(
                (
                    "FaceForensics++",
                    role,
                    dfb_label,
                ),
                0,
            )

            print(
                "  {:11s} {:8s} {:6d}".format(
                    role,
                    dfb_label,
                    count,
                )
            )

    print()
    print(
        "UCF VIEW COUNTS"
    )

    for adapter_dataset in (
        "FF-DF",
        "FF-F2F",
        "FF-FS",
        "FF-NT",
    ):
        videos = sum(
            count
            for key, count in built[
                "video_counts"
            ].items()
            if key[0] == adapter_dataset
        )

        frames = sum(
            count
            for key, count in built[
                "frame_counts"
            ].items()
            if key[0] == adapter_dataset
        )

        print(
            "  {:8s} videos={:4d} frames={:6d}".format(
                adapter_dataset,
                videos,
                frames,
            )
        )

    print()
    print(
        "OUTPUT FILES"
    )

    for adapter_dataset in EXPECTED_ADAPTERS:
        print(
            "  {}".format(
                output_dir
                / "{}.json".format(
                    adapter_dataset
                )
            )
        )

    print(
        "  summary: {}".format(
            summary_path
        )
    )

    print()
    print(
        "TRAINING DATASET JSON BUILD PASSED"
    )


if __name__ == "__main__":
    main()