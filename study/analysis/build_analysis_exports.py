from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from study.analysis.build_analysis_records import (
    read_jsonl,
)


REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


ABSOLUTE_FIELDS = (
    "detector",
    "checkpoint_sha256",
    "dataset",
    "role",
    "condition",
    "group_name",
    "group_type",

    "n_real_nominal",
    "n_manipulated_nominal",
    "n_real_valid",
    "n_manipulated_valid",
    "n_real_invalid",
    "n_manipulated_invalid",

    "auc",
    "auc_ci_lower",
    "auc_ci_upper",

    "fpr",
    "fpr_ci_lower",
    "fpr_ci_upper",

    "fnr",
    "fnr_ci_lower",
    "fnr_ci_upper",

    "hter",
    "hter_ci_lower",
    "hter_ci_upper",

    "tp",
    "fp",
    "tn",
    "fn",
)


PAIRED_FIELDS = (
    "detector",
    "checkpoint_sha256",
    "dataset",
    "role",
    "group_name",
    "group_type",
    "clean_condition",
    "degraded_condition",

    "n_real_clean_valid",
    "n_real_degraded_valid",
    "n_real_paired",

    "n_manipulated_clean_valid",
    "n_manipulated_degraded_valid",
    "n_manipulated_paired",

    "clean_auc",
    "degraded_auc",
    "delta_auc",
    "delta_auc_ci_lower",
    "delta_auc_ci_upper",

    "clean_fpr",
    "degraded_fpr",
    "delta_fpr",
    "delta_fpr_ci_lower",
    "delta_fpr_ci_upper",

    "clean_fnr",
    "degraded_fnr",
    "delta_fnr",
    "delta_fnr_ci_lower",
    "delta_fnr_ci_upper",

    "clean_hter",
    "degraded_hter",
    "delta_hter",
    "delta_hter_ci_lower",
    "delta_hter_ci_upper",
)


DIAGNOSTIC_FIELDS = (
    "detector",
    "checkpoint_sha256",
    "dataset",
    "role",
    "condition",

    "nominal_real_records",
    "valid_real_records",
    "excluded_invalid_real_records",

    "target_fpr",
    "primary_threshold_id",
    "primary_clean_threshold",
    "diagnostic_threshold",
    "threshold_displacement",
    "threshold_displacement_ci_lower",
    "threshold_displacement_ci_upper",

    "diagnostic_realized_fpr",
    "diagnostic_false_positives",
    "diagnostic_max_allowed_false_positives",

    "bootstrap_paired_valid_real",
)


SCORE_FIELDS = (
    "detector",
    "checkpoint_sha256",
    "dataset",
    "role",

    "relative_source_path",
    "pairing_uid",

    "base_video_id",
    "source_video_id",
    "source_group_id",

    "study_label",
    "manipulation",
    "condition",

    "analysis_eligible",
    "analysis_status",

    "target_frame_count",
    "clean_valid_frame_count",
    "valid_frame_count",

    "video_score",

    "threshold_id",
    "threshold_value",
    "decision_status",
    "decision",
)


PROVENANCE_FILES = (
    "study/analysis/analysis_schema.py",
    "study/analysis/build_analysis_records.py",

    "study/analysis/operating_point.py",
    "study/analysis/build_primary_thresholds.py",
    "study/analysis/apply_frozen_thresholds.py",

    "study/analysis/metrics.py",
    "study/analysis/metric_groups.py",
    "study/analysis/paired_metric_groups.py",
    "study/analysis/compute_metrics.py",
    "study/analysis/diagnostic_thresholds.py",
    "study/analysis/build_metric_results.py",

    "study/analysis/bootstrap_sampling.py",
    "study/analysis/bootstrap_metrics.py",
    "study/analysis/bootstrap_views.py",
    "study/analysis/build_bootstrap_results.py",

    "study/analysis/build_analysis_exports.py",

    "training/config/detector/study_xception.yaml",
    "training/config/detector/study_ucf.yaml",
    "training/config/detector/study_spsl.yaml",

    "study/manifests/artifacts/study_manifest.csv",
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


def write_csv(
    path: Path,
    *,
    rows: Iterable[
        Dict[str, Any]
    ],
    fields: Iterable[str],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = list(
        rows
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                fields
            ),
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(
                        field,
                        "",
                    )
                    for field in fields
                }
            )

    temporary.replace(
        path
    )


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
) -> Tuple[
    str,
    str,
]:
    condition = row.get(
        "condition"
    )

    if condition is None:
        condition = row.get(
            "degraded_condition"
        )

    return (
        str(
            row[
                "detector"
            ]
        ),
        str(
            condition
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


def build_absolute_export(
    *,
    metric_results: Dict[str, Any],
    bootstrap_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    point_index = _unique_index(
        metric_results.get(
            "absolute_metrics",
            [],
        ),
        key_function=(
            _absolute_key
        ),
        context=(
            "absolute point result"
        ),
    )

    bootstrap_index = _unique_index(
        bootstrap_results.get(
            "absolute_metrics",
            [],
        ),
        key_function=(
            _absolute_key
        ),
        context=(
            "absolute bootstrap result"
        ),
    )

    if (
        set(
            point_index
        )
        != set(
            bootstrap_index
        )
    ):
        raise ValueError(
            "Absolute point/bootstrap result "
            "keys do not match."
        )

    output = []

    for key in sorted(
        point_index
    ):
        point = point_index[
            key
        ]

        bootstrap = bootstrap_index[
            key
        ]

        row = {
            field: point.get(
                field
            )
            for field in (
                "detector",
                "checkpoint_sha256",
                "dataset",
                "role",
                "condition",
                "group_name",
                "group_type",

                "n_real_nominal",
                "n_manipulated_nominal",
                "n_real_valid",
                "n_manipulated_valid",
                "n_real_invalid",
                "n_manipulated_invalid",

                "auc",
                "fpr",
                "fnr",
                "hter",

                "tp",
                "fp",
                "tn",
                "fn",
            )
        }

        for metric in (
            "auc",
            "fpr",
            "fnr",
            "hter",
        ):
            interval = (
                bootstrap[
                    "confidence_intervals"
                ][
                    metric
                ]
            )

            row[
                "{}_ci_lower".format(
                    metric
                )
            ] = interval[
                "lower"
            ]

            row[
                "{}_ci_upper".format(
                    metric
                )
            ] = interval[
                "upper"
            ]

        output.append(
            row
        )

    return output


def build_paired_export(
    *,
    metric_results: Dict[str, Any],
    bootstrap_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    point_index = _unique_index(
        metric_results.get(
            "paired_degradation_metrics",
            [],
        ),
        key_function=(
            _paired_key
        ),
        context=(
            "paired point result"
        ),
    )

    bootstrap_index = _unique_index(
        bootstrap_results.get(
            "paired_degradation_metrics",
            [],
        ),
        key_function=(
            _paired_key
        ),
        context=(
            "paired bootstrap result"
        ),
    )

    if (
        set(
            point_index
        )
        != set(
            bootstrap_index
        )
    ):
        raise ValueError(
            "Paired point/bootstrap result "
            "keys do not match."
        )

    output = []

    for key in sorted(
        point_index
    ):
        point = point_index[
            key
        ]

        bootstrap = bootstrap_index[
            key
        ]

        row = {
            field: point.get(
                field
            )
            for field in (
                "detector",
                "checkpoint_sha256",
                "dataset",
                "role",
                "group_name",
                "group_type",
                "clean_condition",
                "degraded_condition",

                "n_real_clean_valid",
                "n_real_degraded_valid",
                "n_real_paired",

                "n_manipulated_clean_valid",
                "n_manipulated_degraded_valid",
                "n_manipulated_paired",

                "clean_auc",
                "degraded_auc",
                "delta_auc",

                "clean_fpr",
                "degraded_fpr",
                "delta_fpr",

                "clean_fnr",
                "degraded_fnr",
                "delta_fnr",

                "clean_hter",
                "degraded_hter",
                "delta_hter",
            )
        }

        for metric in (
            "delta_auc",
            "delta_fpr",
            "delta_fnr",
            "delta_hter",
        ):
            interval = (
                bootstrap[
                    "confidence_intervals"
                ][
                    metric
                ]
            )

            row[
                "{}_ci_lower".format(
                    metric
                )
            ] = interval[
                "lower"
            ]

            row[
                "{}_ci_upper".format(
                    metric
                )
            ] = interval[
                "upper"
            ]

        output.append(
            row
        )

    return output


def build_diagnostic_export(
    *,
    metric_results: Dict[str, Any],
    bootstrap_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    point_index = _unique_index(
        metric_results.get(
            "diagnostic_thresholds",
            [],
        ),
        key_function=(
            _diagnostic_key
        ),
        context=(
            "diagnostic point result"
        ),
    )

    bootstrap_index = _unique_index(
        bootstrap_results.get(
            "diagnostic_thresholds",
            [],
        ),
        key_function=(
            _diagnostic_key
        ),
        context=(
            "diagnostic bootstrap result"
        ),
    )

    if (
        set(
            point_index
        )
        != set(
            bootstrap_index
        )
    ):
        raise ValueError(
            "Diagnostic point/bootstrap result "
            "keys do not match."
        )

    output = []

    for key in sorted(
        point_index
    ):
        point = point_index[
            key
        ]

        bootstrap = bootstrap_index[
            key
        ]

        interval = (
            bootstrap[
                "confidence_intervals"
            ][
                "threshold_displacement"
            ]
        )

        row = {
            field: point.get(
                field
            )
            for field in (
                "detector",
                "checkpoint_sha256",
                "dataset",
                "role",
                "condition",

                "nominal_real_records",
                "valid_real_records",
                "excluded_invalid_real_records",

                "target_fpr",
                "primary_threshold_id",
                "primary_clean_threshold",
                "diagnostic_threshold",
                "threshold_displacement",

                "diagnostic_realized_fpr",
                "diagnostic_false_positives",
                "diagnostic_max_allowed_false_positives",
            )
        }

        row[
            "threshold_displacement_ci_lower"
        ] = interval[
            "lower"
        ]

        row[
            "threshold_displacement_ci_upper"
        ] = interval[
            "upper"
        ]

        row[
            "bootstrap_paired_valid_real"
        ] = bootstrap[
            "n_paired_valid_real"
        ]

        output.append(
            row
        )

    return output


def build_score_export(
    decision_records: Iterable[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    rows = []

    seen = set()

    for source in decision_records:
        key = (
            str(
                source[
                    "detector"
                ]
            ),
            str(
                source[
                    "dataset"
                ]
            ),
            str(
                source[
                    "relative_source_path"
                ]
            ),
            str(
                source[
                    "condition"
                ]
            ),
        )

        if key in seen:
            raise ValueError(
                "Duplicate score-record identity: {}"
                .format(
                    key
                )
            )

        seen.add(
            key
        )

        rows.append(
            {
                field: source.get(
                    field
                )
                for field in (
                    SCORE_FIELDS
                )
            }
        )

    rows.sort(
        key=lambda row: (
            str(
                row.get(
                    "detector",
                    "",
                )
            ),
            str(
                row.get(
                    "dataset",
                    "",
                )
            ),
            str(
                row.get(
                    "role",
                    "",
                )
            ),
            str(
                row.get(
                    "relative_source_path",
                    "",
                )
            ),
            str(
                row.get(
                    "condition",
                    "",
                )
            ),
        )
    )

    return rows


def git_provenance() -> Dict[str, Any]:
    def run_git(
        *arguments
    ):
        result = subprocess.run(
            [
                "git",
                *arguments,
            ],
            cwd=str(
                REPO_ROOT
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if (
            result.returncode
            != 0
        ):
            return None

        return result.stdout.strip()

    commit = run_git(
        "rev-parse",
        "HEAD",
    )

    status = run_git(
        "status",
        "--porcelain",
    )

    branch = run_git(
        "rev-parse",
        "--abbrev-ref",
        "HEAD",
    )

    return {
        "commit": (
            commit
        ),

        "branch": (
            branch
        ),

        "dirty": (
            bool(
                status
            )
            if status is not None
            else None
        ),

        "status_available": (
            status is not None
        ),
    }


def tracked_file_hashes():
    output = []

    for relative in (
        PROVENANCE_FILES
    ):
        path = (
            REPO_ROOT
            / relative
        )

        if not path.is_file():
            continue

        output.append(
            {
                "path": (
                    relative
                ),

                "sha256": (
                    sha256_file(
                        path
                    )
                ),

                "size_bytes": (
                    path.stat().st_size
                ),
            }
        )

    return output


def build_provenance(
    *,
    analysis_path: Path,
    decisions_path: Path,
    metric_results_path: Path,
    bootstrap_results_path: Path,
    metric_results: Dict[str, Any],
    bootstrap_results: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,

        "artifact_type": (
            "analysis_provenance"
        ),

        "runtime": {
            "python": (
                sys.version.split()[0]
            ),

            "python_implementation": (
                platform.python_implementation()
            ),

            "platform": (
                platform.platform()
            ),

            "numpy": (
                np.__version__
            ),
        },

        "git": (
            git_provenance()
        ),

        "direct_inputs": {
            "analysis_jsonl": {
                "path": str(
                    analysis_path
                ),

                "sha256": (
                    sha256_file(
                        analysis_path
                    )
                ),
            },

            "decisions_jsonl": {
                "path": str(
                    decisions_path
                ),

                "sha256": (
                    sha256_file(
                        decisions_path
                    )
                ),
            },

            "metric_results_json": {
                "path": str(
                    metric_results_path
                ),

                "sha256": (
                    sha256_file(
                        metric_results_path
                    )
                ),
            },

            "bootstrap_results_json": {
                "path": str(
                    bootstrap_results_path
                ),

                "sha256": (
                    sha256_file(
                        bootstrap_results_path
                    )
                ),
            },
        },

        #
        # Preserve the upstream provenance chains rather
        # than attempting to reconstruct them here.
        #
        "metric_result_sources": (
            metric_results.get(
                "source_artifacts",
                {}
            )
        ),

        "bootstrap_result_sources": (
            bootstrap_results.get(
                "source_artifacts",
                {}
            )
        ),

        "bootstrap_configuration": (
            bootstrap_results.get(
                "bootstrap_configuration"
            )
        ),

        "tracked_files": (
            tracked_file_hashes()
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export Section 5.6 analysis results "
            "into compact CSV/JSON tables and retain "
            "analysis provenance."
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
        "--bootstrap-results-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--allow-empty-primary-metrics",
        action="store_true",
        help=(
            "Permit validation-only smoke input "
            "without held-out absolute or paired "
            "results. Do not use for production."
        ),
    )

    args = parser.parse_args()

    analysis_records = read_jsonl(
        args.analysis_jsonl
    )

    decision_records = read_jsonl(
        args.decisions_jsonl
    )

    metric_results = read_json(
        args.metric_results_json
    )

    bootstrap_results = read_json(
        args.bootstrap_results_json
    )

    if (
        metric_results.get(
            "artifact_type"
        )
        != "metric_results"
    ):
        raise ValueError(
            "Unexpected metric-result artifact."
        )

    if (
        bootstrap_results.get(
            "artifact_type"
        )
        != "bootstrap_results"
    ):
        raise ValueError(
            "Unexpected bootstrap-result artifact."
        )

    absolute_rows = (
        build_absolute_export(
            metric_results=(
                metric_results
            ),
            bootstrap_results=(
                bootstrap_results
            ),
        )
    )

    paired_rows = (
        build_paired_export(
            metric_results=(
                metric_results
            ),
            bootstrap_results=(
                bootstrap_results
            ),
        )
    )

    diagnostic_rows = (
        build_diagnostic_export(
            metric_results=(
                metric_results
            ),
            bootstrap_results=(
                bootstrap_results
            ),
        )
    )

    score_rows = build_score_export(
        decision_records
    )

    if (
        not args.allow_empty_primary_metrics
        and not absolute_rows
    ):
        raise RuntimeError(
            "No held-out absolute results "
            "are available for export."
        )

    if (
        not args.allow_empty_primary_metrics
        and not paired_rows
    ):
        raise RuntimeError(
            "No paired degradation results "
            "are available for export."
        )

    if not diagnostic_rows:
        raise RuntimeError(
            "No diagnostic threshold results "
            "are available for export."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    absolute_path = (
        args.output_dir
        / "absolute_metrics.csv"
    )

    paired_path = (
        args.output_dir
        / "paired_degradation_metrics.csv"
    )

    diagnostic_path = (
        args.output_dir
        / "diagnostic_thresholds.csv"
    )

    score_path = (
        args.output_dir
        / "score_records.csv"
    )

    compact_path = (
        args.output_dir
        / "analysis_results_compact.json"
    )

    provenance_path = (
        args.output_dir
        / "analysis_provenance.json"
    )

    summary_path = (
        args.output_dir
        / "analysis_export_summary.json"
    )

    write_csv(
        absolute_path,
        rows=absolute_rows,
        fields=ABSOLUTE_FIELDS,
    )

    write_csv(
        paired_path,
        rows=paired_rows,
        fields=PAIRED_FIELDS,
    )

    write_csv(
        diagnostic_path,
        rows=diagnostic_rows,
        fields=DIAGNOSTIC_FIELDS,
    )

    write_csv(
        score_path,
        rows=score_rows,
        fields=SCORE_FIELDS,
    )

    compact = {
        "schema_version": 1,

        "artifact_type": (
            "analysis_results_compact"
        ),

        "absolute_metric_count": (
            len(
                absolute_rows
            )
        ),

        "paired_degradation_count": (
            len(
                paired_rows
            )
        ),

        "diagnostic_threshold_count": (
            len(
                diagnostic_rows
            )
        ),

        "score_record_count": (
            len(
                score_rows
            )
        ),

        "absolute_metrics": (
            absolute_rows
        ),

        "paired_degradation_metrics": (
            paired_rows
        ),

        "diagnostic_thresholds": (
            diagnostic_rows
        ),
    }

    write_json(
        compact_path,
        compact,
    )

    provenance = build_provenance(
        analysis_path=(
            args.analysis_jsonl
        ),
        decisions_path=(
            args.decisions_jsonl
        ),
        metric_results_path=(
            args.metric_results_json
        ),
        bootstrap_results_path=(
            args.bootstrap_results_json
        ),
        metric_results=(
            metric_results
        ),
        bootstrap_results=(
            bootstrap_results
        ),
    )

    write_json(
        provenance_path,
        provenance,
    )

    output_files = {
        "absolute_metrics_csv": (
            absolute_path
        ),

        "paired_degradation_metrics_csv": (
            paired_path
        ),

        "diagnostic_thresholds_csv": (
            diagnostic_path
        ),

        "score_records_csv": (
            score_path
        ),

        "compact_results_json": (
            compact_path
        ),

        "analysis_provenance_json": (
            provenance_path
        ),
    }

    summary = {
        "schema_version": 1,

        "artifact_type": (
            "analysis_export_summary"
        ),

        "absolute_metric_count": (
            len(
                absolute_rows
            )
        ),

        "paired_degradation_count": (
            len(
                paired_rows
            )
        ),

        "diagnostic_threshold_count": (
            len(
                diagnostic_rows
            )
        ),

        "score_record_count": (
            len(
                score_rows
            )
        ),

        "allow_empty_primary_metrics": (
            args.allow_empty_primary_metrics
        ),

        "bootstrap_configuration": (
            bootstrap_results[
                "bootstrap_configuration"
            ]
        ),

        "outputs": {
            name: {
                "path": str(
                    path
                ),

                "sha256": (
                    sha256_file(
                        path
                    )
                ),

                "size_bytes": (
                    path.stat().st_size
                ),
            }
            for (
                name,
                path
            ) in output_files.items()
        },

        "status": "passed",
    }

    write_json(
        summary_path,
        summary,
    )

    print(
        "ANALYSIS RESULT EXPORT"
    )
    print(
        "  absolute metrics:          {}"
        .format(
            len(
                absolute_rows
            )
        )
    )
    print(
        "  paired degradation:       {}"
        .format(
            len(
                paired_rows
            )
        )
    )
    print(
        "  diagnostic thresholds:    {}"
        .format(
            len(
                diagnostic_rows
            )
        )
    )
    print(
        "  score records:            {}"
        .format(
            len(
                score_rows
            )
        )
    )
    print(
        "  output directory:         {}"
        .format(
            args.output_dir
        )
    )
    print()
    print(
        "ANALYSIS RESULT EXPORT PASSED"
    )


if __name__ == "__main__":
    main()