from __future__ import annotations

import json
import math
from pathlib import Path

from study.analysis.build_analysis_records import (
    read_jsonl,
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

THRESHOLDS_PATH = (
    STUDY_ROOT
    / "analysis"
    / "primary_thresholds_smoke.json"
)

RESULT_PATH = (
    STUDY_ROOT
    / "analysis"
    / "metric_results_smoke.json"
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

    thresholds = read_json(
        THRESHOLDS_PATH
    )

    results = read_json(
        RESULT_PATH
    )

    assert len(
        analysis
    ) == 18

    assert len(
        decisions
    ) == 18

    assert (
        results[
            "artifact_type"
        ]
        == "metric_results"
    )

    assert (
        results[
            "analysis_record_count"
        ]
        == 18
    )

    assert (
        results[
            "decision_record_count"
        ]
        == 18
    )

    #
    # Current smoke contains FF++ validation only.
    #
    # Therefore primary test/evaluation metric groups
    # must NOT be fabricated from calibration data.
    #
    assert (
        results[
            "absolute_metric_result_count"
        ]
        == 0
    )

    assert (
        results[
            "paired_degradation_result_count"
        ]
        == 0
    )

    assert (
        results[
            "absolute_metrics"
        ]
        == []
    )

    assert (
        results[
            "paired_degradation_metrics"
        ]
        == []
    )

    assert (
        results[
            "primary_metric_population_present"
        ]
        is False
    )

    assert (
        results[
            "paired_degradation_population_present"
        ]
        is False
    )

    #
    # There are three detectors and RSZ is the only
    # processed validation condition in this smoke.
    #
    diagnostics = (
        results[
            "diagnostic_thresholds"
        ]
    )

    assert len(
        diagnostics
    ) == 3

    assert (
        results[
            "diagnostic_threshold_result_count"
        ]
        == 3
    )

    assert {
        row[
            "detector"
        ]
        for row in diagnostics
    } == {
        "xception",
        "ucf",
        "spsl",
    }

    assert {
        row[
            "condition"
        ]
        for row in diagnostics
    } == {
        "RSZ"
    }

    primary_index = {
        row[
            "detector"
        ]: row
        for row in thresholds[
            "thresholds"
        ]
    }

    assert set(
        primary_index
    ) == {
        "xception",
        "ucf",
        "spsl",
    }

    for diagnostic in diagnostics:
        detector = diagnostic[
            "detector"
        ]

        primary = (
            primary_index[
                detector
            ]
        )

        assert (
            diagnostic[
                "checkpoint_sha256"
            ]
            == primary[
                "checkpoint_sha256"
            ]
        )

        assert (
            diagnostic[
                "primary_threshold_id"
            ]
            == primary[
                "threshold_id"
            ]
        )

        assert math.isclose(
            diagnostic[
                "primary_clean_threshold"
            ],
            primary[
                "threshold_value"
            ],
            rel_tol=0.0,
            abs_tol=0.0,
        )

        assert (
            diagnostic[
                "nominal_real_records"
            ]
            == 3
        )

        assert (
            diagnostic[
                "valid_real_records"
            ]
            == 3
        )

        assert (
            diagnostic[
                "diagnostic_max_allowed_false_positives"
            ]
            == 0
        )

        assert (
            diagnostic[
                "diagnostic_false_positives"
            ]
            == 0
        )

        assert math.isclose(
            diagnostic[
                "diagnostic_realized_fpr"
            ],
            0.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )

        assert (
            diagnostic[
                "clean_threshold_reestimated"
            ]
            is False
        )

        assert (
            diagnostic[
                "processed_failure_changes_primary_threshold"
            ]
            is False
        )

    #
    # Decision adapter must preserve the original
    # analysis identities and scores.
    #
    analysis_index = {
        (
            row[
                "detector"
            ],
            row[
                "dataset"
            ],
            row[
                "relative_source_path"
            ],
            row[
                "condition"
            ],
        ): row
        for row in analysis
    }

    decision_index = {
        (
            row[
                "detector"
            ],
            row[
                "dataset"
            ],
            row[
                "relative_source_path"
            ],
            row[
                "condition"
            ],
        ): row
        for row in decisions
    }

    assert (
        set(
            analysis_index
        )
        == set(
            decision_index
        )
    )

    for key in analysis_index:
        assert (
            analysis_index[
                key
            ][
                "video_score"
            ]
            == decision_index[
                key
            ][
                "video_score"
            ]
        )

    print(
        "METRIC RESULT ARTIFACT VALIDATION"
    )
    print(
        "  analysis records:             18"
    )
    print(
        "  decision records:             18"
    )
    print(
        "  held-out absolute metrics:    0 (expected)"
    )
    print(
        "  paired degradation metrics:   0 (expected)"
    )
    print(
        "  diagnostic thresholds:        3"
    )
    print(
        "  validation/test separation:   PASSED"
    )
    print(
        "  detector coverage:            PASSED"
    )
    print(
        "  primary-threshold linkage:    PASSED"
    )
    print(
        "  processed diagnostic FPR:     PASSED"
    )
    print(
        "  frozen clean threshold:       PASSED"
    )
    print(
        "  score/identity preservation:  PASSED"
    )
    print()
    print(
        "METRIC RESULT ARTIFACT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()