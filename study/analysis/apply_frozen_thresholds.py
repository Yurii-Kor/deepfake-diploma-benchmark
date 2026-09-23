from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from study.analysis.build_analysis_records import (
    read_jsonl,
)
from study.analysis.operating_point import (
    apply_frozen_threshold,
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


def read_json(
    path: Path,
) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def build_threshold_index(
    artifact: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    if (
        artifact.get(
            "artifact_type"
        )
        != "primary_operating_thresholds"
    ):
        raise ValueError(
            "Unexpected threshold artifact type."
        )

    thresholds = artifact.get(
        "thresholds"
    )

    if not isinstance(
        thresholds,
        list,
    ):
        raise ValueError(
            "Threshold artifact has no "
            "threshold list."
        )

    index = {}

    for row in thresholds:
        detector = str(
            row[
                "detector"
            ]
        )

        if detector in index:
            raise ValueError(
                "Duplicate threshold for detector: {}"
                .format(
                    detector
                )
            )

        index[
            detector
        ] = row

    if not index:
        raise ValueError(
            "Threshold artifact is empty."
        )

    return index


def build_decision_records(
    *,
    analysis_records: List[
        Dict[str, Any]
    ],
    threshold_artifact: Dict[
        str,
        Any,
    ],
) -> List[Dict[str, Any]]:
    threshold_index = (
        build_threshold_index(
            threshold_artifact
        )
    )

    output = []

    for row in analysis_records:
        detector = str(
            row[
                "detector"
            ]
        )

        threshold = (
            threshold_index.get(
                detector
            )
        )

        if threshold is None:
            raise ValueError(
                "No frozen threshold exists "
                "for detector {}.".format(
                    detector
                )
            )

        output.append(
            apply_frozen_threshold(
                analysis_record=row,
                threshold_record=(
                    threshold
                ),
            )
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

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in records:
            file.write(
                json.dumps(
                    row,
                    sort_keys=True,
                )
            )

            file.write(
                "\n"
            )

    temporary.replace(
        path
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply detector-specific frozen clean "
            "FF++ validation thresholds to canonical "
            "analysis records."
        )
    )

    parser.add_argument(
        "--analysis-jsonl",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--thresholds-json",
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

    analysis_records = read_jsonl(
        args.analysis_jsonl
    )

    if not analysis_records:
        raise RuntimeError(
            "Analysis input is empty."
        )

    threshold_artifact = read_json(
        args.thresholds_json
    )

    decision_records = (
        build_decision_records(
            analysis_records=(
                analysis_records
            ),
            threshold_artifact=(
                threshold_artifact
            ),
        )
    )

    write_jsonl(
        args.output_jsonl,
        decision_records,
    )

    decision_status_counts = Counter(
        row[
            "decision_status"
        ]
        for row in decision_records
    )

    decision_counts = Counter(
        str(
            row[
                "decision"
            ]
        )
        for row in decision_records
        if (
            row[
                "decision"
            ]
            is not None
        )
    )

    thresholds_by_detector = {}

    for row in decision_records:
        detector = row[
            "detector"
        ]

        value = float(
            row[
                "threshold_value"
            ]
        )

        threshold_id = row[
            "threshold_id"
        ]

        previous = (
            thresholds_by_detector.get(
                detector
            )
        )

        current = (
            threshold_id,
            value,
        )

        if (
            previous is not None
            and previous != current
        ):
            raise RuntimeError(
                "Detector {} received more "
                "than one frozen threshold."
                .format(
                    detector
                )
            )

        thresholds_by_detector[
            detector
        ] = current

    summary = {
        "schema_version": 1,
        "artifact_type": (
            "frozen_threshold_decisions"
        ),

        "analysis_record_count": (
            len(
                analysis_records
            )
        ),

        "decision_record_count": (
            len(
                decision_records
            )
        ),

        "decision_status_counts": dict(
            sorted(
                decision_status_counts.items()
            )
        ),

        "decision_counts": dict(
            sorted(
                decision_counts.items()
            )
        ),

        "detector_threshold_count": (
            len(
                thresholds_by_detector
            )
        ),

        "analysis_input_sha256": (
            sha256_file(
                args.analysis_jsonl
            )
        ),

        "threshold_artifact_sha256": (
            sha256_file(
                args.thresholds_json
            )
        ),

        "decision_output_sha256": (
            sha256_file(
                args.output_jsonl
            )
        ),

        "status": "passed",
    }

    write_json(
        args.summary_json,
        summary,
    )

    print(
        "FROZEN THRESHOLD APPLICATION"
    )
    print(
        "  analysis records:   {}".format(
            len(
                analysis_records
            )
        )
    )
    print(
        "  decision records:   {}".format(
            len(
                decision_records
            )
        )
    )
    print(
        "  detector thresholds:{}".format(
            len(
                thresholds_by_detector
            )
        )
    )
    print(
        "  decision statuses:  {}".format(
            dict(
                sorted(
                    decision_status_counts.items()
                )
            )
        )
    )
    print(
        "  output:             {}".format(
            args.output_jsonl
        )
    )
    print(
        "  summary:            {}".format(
            args.summary_json
        )
    )
    print()
    print(
        "FROZEN THRESHOLD APPLICATION PASSED"
    )


if __name__ == "__main__":
    main()