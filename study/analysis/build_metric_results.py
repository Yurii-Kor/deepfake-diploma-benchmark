from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from study.analysis.apply_frozen_thresholds import (
    read_json,
)
from study.analysis.build_analysis_records import (
    read_jsonl,
)
from study.analysis.compute_metrics import (
    compute_absolute_metrics,
    compute_paired_degradation_metrics,
)
from study.analysis.diagnostic_thresholds import (
    compute_diagnostic_thresholds,
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


def analysis_identity(
    row: Dict[str, Any],
) -> Tuple[
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
                "relative_source_path"
            ]
        ),
        str(
            row[
                "condition"
            ]
        ),
    )


def validate_analysis_decision_alignment(
    *,
    analysis_records: List[
        Dict[str, Any]
    ],
    decision_records: List[
        Dict[str, Any]
    ],
):
    analysis_index = {}

    for row in analysis_records:
        key = analysis_identity(
            row
        )

        if key in analysis_index:
            raise ValueError(
                "Duplicate analysis identity: {}"
                .format(
                    key
                )
            )

        analysis_index[
            key
        ] = row

    decision_index = {}

    for row in decision_records:
        key = analysis_identity(
            row
        )

        if key in decision_index:
            raise ValueError(
                "Duplicate decision identity: {}"
                .format(
                    key
                )
            )

        decision_index[
            key
        ] = row

    if (
        set(
            analysis_index
        )
        != set(
            decision_index
        )
    ):
        missing_decisions = (
            set(
                analysis_index
            )
            - set(
                decision_index
            )
        )

        extra_decisions = (
            set(
                decision_index
            )
            - set(
                analysis_index
            )
        )

        raise ValueError(
            "Analysis/decision identity mismatch: "
            "missing_decisions={} extra_decisions={}"
            .format(
                len(
                    missing_decisions
                ),
                len(
                    extra_decisions
                ),
            )
        )

    invariant_fields = (
        "checkpoint_sha256",
        "role",
        "pairing_uid",
        "base_video_id",
        "source_video_id",
        "study_label",
        "manipulation",
        "analysis_eligible",
        "analysis_status",
        "video_score",
    )

    for key in sorted(
        analysis_index
    ):
        analysis_row = (
            analysis_index[
                key
            ]
        )

        decision_row = (
            decision_index[
                key
            ]
        )

        for field in invariant_fields:
            if (
                analysis_row.get(
                    field
                )
                != decision_row.get(
                    field
                )
            ):
                raise ValueError(
                    "Decision artifact changed "
                    "analysis field {} for {}."
                    .format(
                        field,
                        key,
                    )
                )


def build_metric_result_artifact(
    *,
    analysis_records: List[
        Dict[str, Any]
    ],
    decision_records: List[
        Dict[str, Any]
    ],
    primary_threshold_artifact: Dict[
        str,
        Any,
    ],
    allow_empty_primary_metrics: bool = False,
) -> Dict[str, Any]:
    validate_analysis_decision_alignment(
        analysis_records=(
            analysis_records
        ),
        decision_records=(
            decision_records
        ),
    )

    absolute_metrics = (
        compute_absolute_metrics(
            decision_records
        )
    )

    paired_degradation_metrics = (
        compute_paired_degradation_metrics(
            decision_records
        )
    )

    diagnostic_thresholds = (
        compute_diagnostic_thresholds(
            analysis_records=(
                analysis_records
            ),
            primary_threshold_artifact=(
                primary_threshold_artifact
            ),
        )
    )

    if (
        not allow_empty_primary_metrics
        and not absolute_metrics
    ):
        raise RuntimeError(
            "No primary held-out absolute metric "
            "results were produced."
        )

    if (
        not allow_empty_primary_metrics
        and not paired_degradation_metrics
    ):
        raise RuntimeError(
            "No paired degradation metric "
            "results were produced."
        )

    if not diagnostic_thresholds:
        raise RuntimeError(
            "No diagnostic threshold results "
            "were produced."
        )

    detectors = sorted(
        {
            str(
                row[
                    "detector"
                ]
            )
            for row in (
                analysis_records
            )
        }
    )

    return {
        "schema_version": 1,

        "artifact_type": (
            "metric_results"
        ),

        "detectors": (
            detectors
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

        "absolute_metric_result_count": (
            len(
                absolute_metrics
            )
        ),

        "paired_degradation_result_count": (
            len(
                paired_degradation_metrics
            )
        ),

        "diagnostic_threshold_result_count": (
            len(
                diagnostic_thresholds
            )
        ),

        "primary_metric_population_present": (
            bool(
                absolute_metrics
            )
        ),

        "paired_degradation_population_present": (
            bool(
                paired_degradation_metrics
            )
        ),

        "absolute_metrics": (
            absolute_metrics
        ),

        "paired_degradation_metrics": (
            paired_degradation_metrics
        ),

        "diagnostic_thresholds": (
            diagnostic_thresholds
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
            "Build the common 5.6.3 metric-result "
            "artifact from canonical analysis records, "
            "frozen decisions, and primary thresholds."
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
        "--thresholds-json",
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
        "--allow-empty-primary-metrics",
        action="store_true",
        help=(
            "Permit validation-only smoke input "
            "without held-out evaluation metric groups. "
            "Do not use for production analysis."
        ),
    )

    args = parser.parse_args()

    analysis_records = read_jsonl(
        args.analysis_jsonl
    )

    decision_records = read_jsonl(
        args.decisions_jsonl
    )

    primary_threshold_artifact = (
        read_json(
            args.thresholds_json
        )
    )

    if not analysis_records:
        raise RuntimeError(
            "Analysis input is empty."
        )

    if not decision_records:
        raise RuntimeError(
            "Decision input is empty."
        )

    artifact = (
        build_metric_result_artifact(
            analysis_records=(
                analysis_records
            ),
            decision_records=(
                decision_records
            ),
            primary_threshold_artifact=(
                primary_threshold_artifact
            ),
            allow_empty_primary_metrics=(
                args.allow_empty_primary_metrics
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

        "primary_thresholds_json": {
            "path": str(
                args.thresholds_json
            ),
            "sha256": (
                sha256_file(
                    args.thresholds_json
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
            "metric_results_summary"
        ),

        "analysis_record_count": (
            artifact[
                "analysis_record_count"
            ]
        ),

        "decision_record_count": (
            artifact[
                "decision_record_count"
            ]
        ),

        "absolute_metric_result_count": (
            artifact[
                "absolute_metric_result_count"
            ]
        ),

        "paired_degradation_result_count": (
            artifact[
                "paired_degradation_result_count"
            ]
        ),

        "diagnostic_threshold_result_count": (
            artifact[
                "diagnostic_threshold_result_count"
            ]
        ),

        "primary_metric_population_present": (
            artifact[
                "primary_metric_population_present"
            ]
        ),

        "paired_degradation_population_present": (
            artifact[
                "paired_degradation_population_present"
            ]
        ),

        "detectors": (
            artifact[
                "detectors"
            ]
        ),

        "allow_empty_primary_metrics": (
            args.allow_empty_primary_metrics
        ),

        "metric_results_json": {
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
        "METRIC RESULT ARTIFACT BUILD"
    )
    print(
        "  analysis records:          {}"
        .format(
            artifact[
                "analysis_record_count"
            ]
        )
    )
    print(
        "  decision records:          {}"
        .format(
            artifact[
                "decision_record_count"
            ]
        )
    )
    print(
        "  absolute metric results:   {}"
        .format(
            artifact[
                "absolute_metric_result_count"
            ]
        )
    )
    print(
        "  paired degradation results:{}"
        .format(
            artifact[
                "paired_degradation_result_count"
            ]
        )
    )
    print(
        "  diagnostic thresholds:     {}"
        .format(
            artifact[
                "diagnostic_threshold_result_count"
            ]
        )
    )
    print(
        "  output:                    {}"
        .format(
            args.output_json
        )
    )
    print(
        "  summary:                   {}"
        .format(
            args.summary_json
        )
    )
    print()
    print(
        "METRIC RESULT ARTIFACT BUILD PASSED"
    )


if __name__ == "__main__":
    main()