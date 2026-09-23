from __future__ import annotations

import json
import math
from pathlib import Path

from study.analysis.bootstrap_sampling import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
)
from study.analysis.build_analysis_records import (
    read_jsonl,
)
from study.analysis.build_bootstrap_results import (
    build_bootstrap_result_artifact,
)


STUDY_ROOT = (
    Path.home()
    / "deepfake_lab"
    / "study_data"
)

ANALYSIS_PATH = (
    STUDY_ROOT
    / "analysis"
    / "analysis_records_smoke.jsonl"
)

DECISIONS_PATH = (
    STUDY_ROOT
    / "analysis"
    / "decisions_smoke.jsonl"
)

METRIC_RESULTS_PATH = (
    STUDY_ROOT
    / "analysis"
    / "metric_results_smoke.json"
)

BOOTSTRAP_RESULTS_PATH = (
    STUDY_ROOT
    / "analysis"
    / "bootstrap_results_smoke.json"
)


def read_json(
    path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def main():
    analysis = read_jsonl(
        ANALYSIS_PATH
    )

    decisions = read_jsonl(
        DECISIONS_PATH
    )

    metric_results = read_json(
        METRIC_RESULTS_PATH
    )

    bootstrap = read_json(
        BOOTSTRAP_RESULTS_PATH
    )

    assert (
        bootstrap[
            "artifact_type"
        ]
        == "bootstrap_results"
    )

    config = (
        bootstrap[
            "bootstrap_configuration"
        ]
    )

    assert (
        config[
            "replicates"
        ]
        == BOOTSTRAP_REPLICATES
        == 2000
    )

    assert math.isclose(
        config[
            "ci_level"
        ],
        BOOTSTRAP_CI_LEVEL,
        rel_tol=0.0,
        abs_tol=0.0,
    )

    assert (
        config[
            "seed"
        ]
        == BOOTSTRAP_SEED
        == 1024
    )

    assert (
        config[
            "ci_method"
        ]
        == "percentile_linear_interpolation"
    )

    assert (
        config[
            "threshold_reestimated_each_replicate"
        ]
        is True
    )

    assert (
        config[
            "paired_clean_degraded_sampling"
        ]
        is True
    )

    assert (
        config[
            "production_configuration"
        ]
        is True
    )

    assert {
        *bootstrap[
            "detectors"
        ]
    } == {
        "spsl",
        "ucf",
        "xception",
    }

    #
    # Validation-only smoke must still not fabricate
    # held-out evaluation bootstrap results.
    #
    assert (
        bootstrap[
            "absolute_metric_bootstrap_count"
        ]
        == 0
    )

    assert (
        bootstrap[
            "paired_degradation_bootstrap_count"
        ]
        == 0
    )

    assert (
        bootstrap[
            "absolute_metrics"
        ]
        == []
    )

    assert (
        bootstrap[
            "paired_degradation_metrics"
        ]
        == []
    )

    #
    # One clean-threshold bootstrap distribution per
    # detector.
    #
    threshold_rows = (
        bootstrap[
            "clean_threshold_bootstrap"
        ]
    )

    assert len(
        threshold_rows
    ) == 3

    assert {
        row[
            "detector"
        ]
        for row in threshold_rows
    } == {
        "spsl",
        "ucf",
        "xception",
    }

    for row in threshold_rows:
        assert (
            row[
                "valid_clean_validation_real"
            ]
            == 3
        )

        assert (
            row[
                "provenance_only"
            ]
            is True
        )

        assert (
            len(
                row[
                    "bootstrap_values"
                ]
            )
            == 2000
        )

        interval = (
            row[
                "percentile_interval"
            ]
        )

        assert (
            interval[
                "replicate_count"
            ]
            == 2000
        )

        assert (
            interval[
                "lower"
            ]
            <= interval[
                "upper"
            ]
        )

        assert math.isfinite(
            interval[
                "lower"
            ]
        )

        assert math.isfinite(
            interval[
                "upper"
            ]
        )

    #
    # RSZ diagnostic bootstrap for three detectors.
    #
    diagnostic_rows = (
        bootstrap[
            "diagnostic_thresholds"
        ]
    )

    assert (
        bootstrap[
            "diagnostic_threshold_bootstrap_count"
        ]
        == 3
    )

    assert len(
        diagnostic_rows
    ) == 3

    assert {
        row[
            "detector"
        ]
        for row in diagnostic_rows
    } == {
        "spsl",
        "ucf",
        "xception",
    }

    assert {
        row[
            "degraded_condition"
        ]
        for row in diagnostic_rows
    } == {
        "RSZ"
    }

    for row in diagnostic_rows:
        assert (
            row[
                "n_clean_valid_real"
            ]
            == 3
        )

        assert (
            row[
                "n_degraded_valid_real"
            ]
            == 3
        )

        assert (
            row[
                "n_paired_valid_real"
            ]
            == 3
        )

        values = (
            row[
                "bootstrap_values"
            ]
        )

        assert (
            len(
                values[
                    "clean_threshold"
                ]
            )
            == 2000
        )

        assert (
            len(
                values[
                    "diagnostic_threshold"
                ]
            )
            == 2000
        )

        assert (
            len(
                values[
                    "threshold_displacement"
                ]
            )
            == 2000
        )

        interval = (
            row[
                "confidence_intervals"
            ][
                "threshold_displacement"
            ]
        )

        assert (
            interval[
                "replicate_count"
            ]
            == 2000
        )

        assert (
            interval[
                "lower"
            ]
            <= interval[
                "upper"
            ]
        )

        assert math.isfinite(
            interval[
                "lower"
            ]
        )

        assert math.isfinite(
            interval[
                "upper"
            ]
        )

    #
    # Rebuild the scientific core and require exact
    # deterministic equality.
    #
    rebuilt = (
        build_bootstrap_result_artifact(
            analysis_records=(
                analysis
            ),
            decision_records=(
                decisions
            ),
            metric_result_artifact=(
                metric_results
            ),
            replicates=(
                BOOTSTRAP_REPLICATES
            ),
            allow_empty_primary_metrics=True,
        )
    )

    core_keys = (
        "schema_version",
        "artifact_type",
        "bootstrap_configuration",
        "detectors",
        "clean_threshold_bootstrap",
        "absolute_metric_bootstrap_count",
        "paired_degradation_bootstrap_count",
        "diagnostic_threshold_bootstrap_count",
        "absolute_metrics",
        "paired_degradation_metrics",
        "diagnostic_thresholds",
    )

    for key in core_keys:
        assert (
            bootstrap[
                key
            ]
            == rebuilt[
                key
            ]
        )

    print(
        "BOOTSTRAP RESULT ARTIFACT VALIDATION"
    )
    print(
        "  bootstrap replicates:         2000"
    )
    print(
        "  CI level:                     95%"
    )
    print(
        "  bootstrap seed:               1024"
    )
    print(
        "  detector threshold series:    3"
    )
    print(
        "  held-out absolute results:    0 (expected)"
    )
    print(
        "  paired degradation results:   0 (expected)"
    )
    print(
        "  diagnostic bootstrap results: 3"
    )
    print(
        "  threshold re-estimation:      PASSED"
    )
    print(
        "  paired diagnostic IDs:        PASSED"
    )
    print(
        "  percentile intervals:         PASSED"
    )
    print(
        "  validation/test separation:   PASSED"
    )
    print(
        "  deterministic rebuild:        PASSED"
    )
    print()
    print(
        "BOOTSTRAP RESULT ARTIFACT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()