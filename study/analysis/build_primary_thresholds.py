from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from study.analysis.build_analysis_records import (
    read_jsonl,
)
from study.analysis.operating_point import (
    PRIMARY_TARGET_FPR,
    select_primary_thresholds_from_analysis_records,
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


def build_threshold_artifact(
    *,
    analysis_records: List[
        Dict[str, Any]
    ],
    analysis_input_path: Path,
) -> Dict[str, Any]:
    thresholds = (
        select_primary_thresholds_from_analysis_records(
            analysis_records,
            target_fpr=(
                PRIMARY_TARGET_FPR
            ),
        )
    )

    thresholds.sort(
        key=lambda row: row[
            "detector"
        ]
    )

    detector_names = [
        row[
            "detector"
        ]
        for row in thresholds
    ]

    if (
        len(
            detector_names
        )
        != len(
            set(
                detector_names
            )
        )
    ):
        raise RuntimeError(
            "Duplicate detector threshold."
        )

    return {
        "schema_version": 1,

        "artifact_type": (
            "primary_operating_thresholds"
        ),

        "target_fpr": (
            PRIMARY_TARGET_FPR
        ),

        "threshold_count": (
            len(
                thresholds
            )
        ),

        "analysis_input": {
            "path": str(
                analysis_input_path
            ),
            "sha256": (
                sha256_file(
                    analysis_input_path
                )
            ),
        },

        "thresholds": (
            thresholds
        ),
    }


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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build detector-specific primary "
            "operating thresholds from clean "
            "FaceForensics++ validation real-video "
            "analysis records."
        )
    )

    parser.add_argument(
        "--analysis-jsonl",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    analysis_records = read_jsonl(
        args.analysis_jsonl
    )

    if not analysis_records:
        raise RuntimeError(
            "Analysis input is empty."
        )

    artifact = (
        build_threshold_artifact(
            analysis_records=(
                analysis_records
            ),
            analysis_input_path=(
                args.analysis_jsonl
            ),
        )
    )

    write_json(
        args.output_json,
        artifact,
    )

    print(
        "PRIMARY THRESHOLD ARTIFACT"
    )
    print(
        "  target FPR:       {}".format(
            artifact[
                "target_fpr"
            ]
        )
    )
    print(
        "  thresholds:       {}".format(
            artifact[
                "threshold_count"
            ]
        )
    )

    for row in artifact[
        "thresholds"
    ]:
        print()
        print(
            "  {}".format(
                row[
                    "detector"
                ]
            )
        )
        print(
            "    valid real videos: {} / {}"
            .format(
                row[
                    "valid_real_records"
                ],
                row[
                    "nominal_real_records"
                ],
            )
        )
        print(
            "    threshold:        {:.17g}"
            .format(
                row[
                    "threshold_value"
                ]
            )
        )
        print(
            "    realized FPR:     {:.17g}"
            .format(
                row[
                    "realized_fpr"
                ]
            )
        )
        print(
            "    false positives:  {}"
            .format(
                row[
                    "false_positives"
                ]
            )
        )

    print()
    print(
        "  output:           {}".format(
            args.output_json
        )
    )
    print()
    print(
        "PRIMARY THRESHOLD ARTIFACT PASSED"
    )


if __name__ == "__main__":
    main()