from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


STUDY_ROOT = (
    Path.home()
    / "deepfake_lab"
    / "study_data"
)

ANALYSIS_DIR = (
    STUDY_ROOT
    / "analysis"
)

EXPORT_DIR = (
    ANALYSIS_DIR
    / "exports_smoke"
)

ANALYSIS_PATH = (
    ANALYSIS_DIR
    / "analysis_records_smoke.jsonl"
)

DECISIONS_PATH = (
    ANALYSIS_DIR
    / "decisions_smoke.jsonl"
)

METRIC_RESULTS_PATH = (
    ANALYSIS_DIR
    / "metric_results_smoke.json"
)

BOOTSTRAP_RESULTS_PATH = (
    ANALYSIS_DIR
    / "bootstrap_results_smoke.json"
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
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def read_csv(
    path: Path,
):
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


def main():
    absolute_path = (
        EXPORT_DIR
        / "absolute_metrics.csv"
    )

    paired_path = (
        EXPORT_DIR
        / "paired_degradation_metrics.csv"
    )

    diagnostic_path = (
        EXPORT_DIR
        / "diagnostic_thresholds.csv"
    )

    score_path = (
        EXPORT_DIR
        / "score_records.csv"
    )

    compact_path = (
        EXPORT_DIR
        / "analysis_results_compact.json"
    )

    provenance_path = (
        EXPORT_DIR
        / "analysis_provenance.json"
    )

    summary_path = (
        EXPORT_DIR
        / "analysis_export_summary.json"
    )

    required_paths = (
        absolute_path,
        paired_path,
        diagnostic_path,
        score_path,
        compact_path,
        provenance_path,
        summary_path,
    )

    for path in required_paths:
        assert path.is_file(), (
            "Missing export artifact: {}"
            .format(
                path
            )
        )

    absolute = read_csv(
        absolute_path
    )

    paired = read_csv(
        paired_path
    )

    diagnostic = read_csv(
        diagnostic_path
    )

    scores = read_csv(
        score_path
    )

    compact = read_json(
        compact_path
    )

    provenance = read_json(
        provenance_path
    )

    summary = read_json(
        summary_path
    )

    #
    # Validation-only smoke:
    #
    # no held-out test metrics must be fabricated.
    #
    assert len(
        absolute
    ) == 0

    assert len(
        paired
    ) == 0

    assert len(
        diagnostic
    ) == 3

    assert len(
        scores
    ) == 18

    assert (
        compact[
            "absolute_metric_count"
        ]
        == 0
    )

    assert (
        compact[
            "paired_degradation_count"
        ]
        == 0
    )

    assert (
        compact[
            "diagnostic_threshold_count"
        ]
        == 3
    )

    assert (
        compact[
            "score_record_count"
        ]
        == 18
    )

    #
    # Diagnostic rows must cover exactly the three
    # study detectors and the only processed smoke
    # condition, RSZ.
    #
    assert {
        row[
            "detector"
        ]
        for row in diagnostic
    } == {
        "spsl",
        "ucf",
        "xception",
    }

    assert {
        row[
            "condition"
        ]
        for row in diagnostic
    } == {
        "RSZ"
    }

    for row in diagnostic:
        assert (
            int(
                row[
                    "nominal_real_records"
                ]
            )
            == 3
        )

        assert (
            int(
                row[
                    "valid_real_records"
                ]
            )
            == 3
        )

        assert (
            int(
                row[
                    "bootstrap_paired_valid_real"
                ]
            )
            == 3
        )

        lower = float(
            row[
                "threshold_displacement_ci_lower"
            ]
        )

        upper = float(
            row[
                "threshold_displacement_ci_upper"
            ]
        )

        assert math.isfinite(
            lower
        )

        assert math.isfinite(
            upper
        )

        assert (
            lower
            <= upper
        )

    #
    # Score export preserves all detector-condition
    # records and does not silently remove validation.
    #
    assert {
        row[
            "detector"
        ]
        for row in scores
    } == {
        "spsl",
        "ucf",
        "xception",
    }

    assert {
        row[
            "condition"
        ]
        for row in scores
    } == {
        "CLN",
        "RSZ",
    }

    assert {
        row[
            "role"
        ]
        for row in scores
    } == {
        "validation"
    }

    #
    # Bootstrap configuration must be carried into the
    # export summary unchanged.
    #
    bootstrap_config = (
        summary[
            "bootstrap_configuration"
        ]
    )

    assert (
        bootstrap_config[
            "replicates"
        ]
        == 2000
    )

    assert math.isclose(
        bootstrap_config[
            "ci_level"
        ],
        0.95,
        rel_tol=0.0,
        abs_tol=0.0,
    )

    assert (
        bootstrap_config[
            "seed"
        ]
        == 1024
    )

    assert (
        bootstrap_config[
            "threshold_reestimated_each_replicate"
        ]
        is True
    )

    #
    # Direct input provenance.
    #
    assert (
        provenance[
            "artifact_type"
        ]
        == "analysis_provenance"
    )

    direct = (
        provenance[
            "direct_inputs"
        ]
    )

    expected_inputs = {
        "analysis_jsonl": (
            ANALYSIS_PATH
        ),

        "decisions_jsonl": (
            DECISIONS_PATH
        ),

        "metric_results_json": (
            METRIC_RESULTS_PATH
        ),

        "bootstrap_results_json": (
            BOOTSTRAP_RESULTS_PATH
        ),
    }

    for name, path in (
        expected_inputs.items()
    ):
        assert (
            direct[
                name
            ][
                "sha256"
            ]
            == sha256_file(
                path
            )
        )

    #
    # Runtime and code/config provenance are present.
    #
    assert (
        provenance[
            "runtime"
        ][
            "python"
        ]
    )

    assert (
        provenance[
            "runtime"
        ][
            "numpy"
        ]
    )

    assert (
        len(
            provenance[
                "tracked_files"
            ]
        )
        > 0
    )

    tracked_paths = {
        row[
            "path"
        ]
        for row in (
            provenance[
                "tracked_files"
            ]
        )
    }

    assert (
        "study/analysis/bootstrap_sampling.py"
        in tracked_paths
    )

    assert (
        "study/analysis/build_bootstrap_results.py"
        in tracked_paths
    )

    assert (
        "study/analysis/build_analysis_exports.py"
        in tracked_paths
    )

    #
    # Every output hash recorded in summary must match
    # the actual file.
    #
    for output in (
        summary[
            "outputs"
        ].values()
    ):
        path = Path(
            output[
                "path"
            ]
        )

        assert (
            output[
                "sha256"
            ]
            == sha256_file(
                path
            )
        )

    assert (
        summary[
            "status"
        ]
        == "passed"
    )

    print(
        "ANALYSIS EXPORT VALIDATION"
    )
    print(
        "  absolute CSV rows:             0 (expected)"
    )
    print(
        "  paired CSV rows:               0 (expected)"
    )
    print(
        "  diagnostic CSV rows:           3"
    )
    print(
        "  score-record rows:             18"
    )
    print(
        "  validation/test separation:    PASSED"
    )
    print(
        "  diagnostic CI export:          PASSED"
    )
    print(
        "  bootstrap configuration:       PASSED"
    )
    print(
        "  direct-input hashes:           PASSED"
    )
    print(
        "  code/config hashes:            PASSED"
    )
    print(
        "  runtime provenance:            PASSED"
    )
    print(
        "  output hashes:                 PASSED"
    )
    print()
    print(
        "ANALYSIS EXPORT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()