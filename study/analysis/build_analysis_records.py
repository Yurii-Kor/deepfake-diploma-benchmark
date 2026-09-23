from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from study.analysis.analysis_schema import (
    SUPPORTED_CONDITIONS,
    analysis_key,
    build_analysis_record,
    manifest_key,
)


CONDITION_ORDER = {
    "CLN": 0,
    "RSZ": 1,
    "BLR": 2,
    "H40": 3,
    "PLT": 4,
}


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


def read_jsonl(
    path: Path,
) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(
                        line
                    )
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Invalid JSON at {} line {}."
                    .format(
                        path,
                        line_number,
                    )
                ) from exc

    return records


def read_csv(
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
        return list(
            csv.DictReader(
                file
            )
        )


def build_manifest_index(
    manifest_rows: List[
        Dict[str, str]
    ],
) -> Dict[
    Tuple[str, str],
    Dict[str, str],
]:
    index = {}

    for row in manifest_rows:
        key = manifest_key(
            row
        )

        if key in index:
            raise ValueError(
                "Duplicate canonical manifest "
                "identity: {}".format(
                    key
                )
            )

        index[
            key
        ] = row

    return index


def build_records(
    *,
    inference_rows: List[
        Dict[str, Any]
    ],
    manifest_index: Dict[
        Tuple[str, str],
        Dict[str, str],
    ],
) -> List[Dict[str, Any]]:
    output = []
    seen_analysis_keys = set()

    for inference_row in (
        inference_rows
    ):
        key = manifest_key(
            inference_row
        )

        manifest_row = (
            manifest_index.get(
                key
            )
        )

        if manifest_row is None:
            raise ValueError(
                "Inference record is absent "
                "from canonical study manifest: {}"
                .format(
                    key
                )
            )

        record = (
            build_analysis_record(
                inference_row=(
                    inference_row
                ),
                manifest_row=(
                    manifest_row
                ),
            )
        )

        record_key = analysis_key(
            record
        )

        if (
            record_key
            in seen_analysis_keys
        ):
            raise ValueError(
                "Duplicate analysis record: {}"
                .format(
                    record_key
                )
            )

        seen_analysis_keys.add(
            record_key
        )

        output.append(
            record
        )

    output.sort(
        key=lambda row: (
            row[
                "detector"
            ],
            row[
                "dataset"
            ],
            row[
                "role"
            ],
            row[
                "relative_source_path"
            ],
            CONDITION_ORDER[
                row[
                    "condition"
                ]
            ],
        )
    )

    return output


def write_jsonl(
    path: Path,
    records: List[
        Dict[str, Any]
    ],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.with_name(
            path.name
            + ".tmp"
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    sort_keys=True,
                )
            )

            file.write(
                "\n"
            )

    temporary_path.replace(
        path
    )


def build_summary(
    *,
    records: List[
        Dict[str, Any]
    ],
    input_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    detector_counts = Counter(
        row[
            "detector"
        ]
        for row in records
    )

    dataset_role_counts = Counter(
        (
            row[
                "dataset"
            ],
            row[
                "role"
            ],
        )
        for row in records
    )

    condition_counts = Counter(
        row[
            "condition"
        ]
        for row in records
    )

    status_counts = Counter(
        row[
            "analysis_status"
        ]
        for row in records
    )

    inference_role_mismatches = sum(
        1
        for row in records
        if (
            row[
                "inference_role"
            ]
            != row[
                "role"
            ]
        )
    )

    unique_keys = {
        analysis_key(
            row
        )
        for row in records
    }

    unique_video_uids = {
        row[
            "video_uid"
        ]
        for row in records
    }

    unique_pairing_uids = {
        row[
            "pairing_uid"
        ]
        for row in records
    }

    unique_bootstrap_clusters = {
        row[
            "bootstrap_cluster_uid"
        ]
        for row in records
    }

    complete_target_count = sum(
        1
        for row in records
        if row[
            "complete_target_frame_set"
        ]
    )

    return {
        "schema_version": 1,

        "input_inference_records": (
            len(
                records
            )
        ),

        "analysis_records": (
            len(
                records
            )
        ),

        "unique_analysis_keys": (
            len(
                unique_keys
            )
        ),

        "unique_video_uids": (
            len(
                unique_video_uids
            )
        ),

        "unique_pairing_uids": (
            len(
                unique_pairing_uids
            )
        ),

        "unique_bootstrap_clusters": (
            len(
                unique_bootstrap_clusters
            )
        ),

        "detector_counts": {
            key: value
            for key, value in sorted(
                detector_counts.items()
            )
        },

        "dataset_role_counts": {
            "{}|{}".format(
                dataset,
                role,
            ): value
            for (
                dataset,
                role,
            ), value in sorted(
                dataset_role_counts.items()
            )
        },

        "condition_counts": {
            key: value
            for key, value in sorted(
                condition_counts.items()
            )
        },

        "analysis_status_counts": {
            key: value
            for key, value in sorted(
                status_counts.items()
            )
        },

        "inference_role_mismatch_count": (
            inference_role_mismatches
        ),

        "complete_target_frame_record_count": (
            complete_target_count
        ),

        "supported_conditions": sorted(
            SUPPORTED_CONDITIONS
        ),

        "artifacts": {
            "inference_jsonl": {
                "path": str(
                    input_path
                ),
                "sha256": (
                    sha256_file(
                        input_path
                    )
                ),
            },

            "study_manifest": {
                "path": str(
                    manifest_path
                ),
                "sha256": (
                    sha256_file(
                        manifest_path
                    )
                ),
            },

            "analysis_jsonl": {
                "path": str(
                    output_path
                ),
                "sha256": (
                    sha256_file(
                        output_path
                    )
                ),
            },
        },

        "status": "passed",
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert canonical inference records "
            "into canonical study-analysis records."
        )
    )

    parser.add_argument(
        "--input-jsonl",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--study-manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-jsonl",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--summary-json",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    inference_rows = read_jsonl(
        args.input_jsonl
    )

    if not inference_rows:
        raise RuntimeError(
            "Inference JSONL is empty."
        )

    manifest_rows = read_csv(
        args.study_manifest
    )

    if not manifest_rows:
        raise RuntimeError(
            "Study manifest is empty."
        )

    manifest_index = (
        build_manifest_index(
            manifest_rows
        )
    )

    analysis_records = (
        build_records(
            inference_rows=(
                inference_rows
            ),
            manifest_index=(
                manifest_index
            ),
        )
    )

    write_jsonl(
        args.output_jsonl,
        analysis_records,
    )

    summary = build_summary(
        records=analysis_records,
        input_path=args.input_jsonl,
        manifest_path=(
            args.study_manifest
        ),
        output_path=(
            args.output_jsonl
        ),
    )

    args.summary_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_summary = (
        args.summary_json.with_name(
            args.summary_json.name
            + ".tmp"
        )
    )

    with temporary_summary.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write(
            "\n"
        )

    temporary_summary.replace(
        args.summary_json
    )

    print(
        "ANALYSIS RECORD BUILD"
    )
    print(
        "  input inference records: {}".format(
            len(
                inference_rows
            )
        )
    )
    print(
        "  analysis records:        {}".format(
            len(
                analysis_records
            )
        )
    )
    print(
        "  unique analysis keys:    {}".format(
            summary[
                "unique_analysis_keys"
            ]
        )
    )
    print(
        "  unique video UIDs:       {}".format(
            summary[
                "unique_video_uids"
            ]
        )
    )
    print(
        "  bootstrap clusters:      {}".format(
            summary[
                "unique_bootstrap_clusters"
            ]
        )
    )
    print(
        "  role mismatches:         {}".format(
            summary[
                "inference_role_mismatch_count"
            ]
        )
    )
    print(
        "  output:                  {}".format(
            args.output_jsonl
        )
    )
    print(
        "  summary:                 {}".format(
            args.summary_json
        )
    )
    print()
    print(
        "ANALYSIS RECORD BUILD PASSED"
    )


if __name__ == "__main__":
    main()