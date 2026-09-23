from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from study.analysis.build_analysis_records import (
    read_jsonl,
)
from study.analysis.operating_point import (
    PRIMARY_TARGET_FPR,
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

THRESHOLD_PATH = (
    STUDY_ROOT
    / "analysis"
    / "primary_thresholds_smoke.json"
)

DECISION_PATH = (
    STUDY_ROOT
    / "analysis"
    / "decisions_smoke.jsonl"
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

    thresholds = read_json(
        THRESHOLD_PATH
    )

    decisions = read_jsonl(
        DECISION_PATH
    )

    assert len(
        analysis
    ) == 18

    assert len(
        decisions
    ) == 18

    assert (
        thresholds[
            "target_fpr"
        ]
        == PRIMARY_TARGET_FPR
    )

    threshold_rows = (
        thresholds[
            "thresholds"
        ]
    )

    assert len(
        threshold_rows
    ) == 3

    threshold_index = {
        row[
            "detector"
        ]: row
        for row in threshold_rows
    }

    assert set(
        threshold_index
    ) == {
        "xception",
        "ucf",
        "spsl",
    }

    #
    # Smoke contains only three clean validation
    # real videos per detector. Therefore floor(
    # 0.05 * 3) = 0 false positives.
    #
    cln_scores = defaultdict(
        list
    )

    for row in analysis:
        if (
            row[
                "dataset"
            ]
            == "FaceForensics++"
            and row[
                "role"
            ]
            == "validation"
            and row[
                "condition"
            ]
            == "CLN"
            and row[
                "study_label"
            ]
            == 0
            and row[
                "analysis_eligible"
            ]
            is True
        ):
            cln_scores[
                row[
                    "detector"
                ]
            ].append(
                float(
                    row[
                        "video_score"
                    ]
                )
            )

    for detector, row in (
        threshold_index.items()
    ):
        assert (
            row[
                "nominal_real_records"
            ]
            == 3
        )

        assert (
            row[
                "valid_real_records"
            ]
            == 3
        )

        assert (
            row[
                "max_allowed_false_positives"
            ]
            == 0
        )

        assert (
            row[
                "false_positives"
            ]
            == 0
        )

        assert math.isclose(
            row[
                "realized_fpr"
            ],
            0.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )

        assert (
            float(
                row[
                    "threshold_value"
                ]
            )
            > max(
                cln_scores[
                    detector
                ]
            )
        )

    #
    # Score records are preserved exactly.
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

    assert len(
        analysis_index
    ) == 18

    for row in decisions:
        key = (
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
        )

        original = (
            analysis_index[
                key
            ]
        )

        assert (
            row[
                "video_score"
            ]
            == original[
                "video_score"
            ]
        )

        threshold = (
            threshold_index[
                row[
                    "detector"
                ]
            ]
        )

        assert (
            row[
                "threshold_id"
            ]
            == threshold[
                "threshold_id"
            ]
        )

        assert (
            row[
                "threshold_value"
            ]
            == threshold[
                "threshold_value"
            ]
        )

        assert (
            row[
                "threshold_source_dataset"
            ]
            == "FaceForensics++"
        )

        assert (
            row[
                "threshold_source_role"
            ]
            == "validation"
        )

        assert (
            row[
                "threshold_source_condition"
            ]
            == "CLN"
        )

        assert (
            row[
                "decision_status"
            ]
            == "decided"
        )

    #
    # Exactly one frozen threshold per detector,
    # unchanged between CLN and RSZ.
    #
    applied = defaultdict(
        set
    )

    conditions = defaultdict(
        set
    )

    for row in decisions:
        detector = row[
            "detector"
        ]

        applied[
            detector
        ].add(
            (
                row[
                    "threshold_id"
                ],
                float(
                    row[
                        "threshold_value"
                    ]
                ),
            )
        )

        conditions[
            detector
        ].add(
            row[
                "condition"
            ]
        )

    for detector in (
        "xception",
        "ucf",
        "spsl",
    ):
        assert len(
            applied[
                detector
            ]
        ) == 1

        assert (
            conditions[
                detector
            ]
            == {
                "CLN",
                "RSZ",
            }
        )

    #
    # Since threshold was estimated from the three
    # smoke CLN real records with zero allowable FPs,
    # all three CLN decisions per detector must be real.
    #
    for detector in (
        "xception",
        "ucf",
        "spsl",
    ):
        clean_rows = [
            row
            for row in decisions
            if (
                row[
                    "detector"
                ]
                == detector
                and row[
                    "condition"
                ]
                == "CLN"
            )
        ]

        assert len(
            clean_rows
        ) == 3

        assert all(
            row[
                "decision"
            ]
            == 0
            for row in clean_rows
        )

    print(
        "OPERATING-POINT ARTIFACT VALIDATION"
    )
    print(
        "  threshold artifacts:          3"
    )
    print(
        "  calibration videos/detector:  3"
    )
    print(
        "  smoke max false positives:    0"
    )
    print(
        "  clean realized FPR:           0"
    )
    print(
        "  decision records:             18"
    )
    print(
        "  score preservation:           PASSED"
    )
    print(
        "  threshold provenance:         PASSED"
    )
    print(
        "  one threshold/detector:       PASSED"
    )
    print(
        "  CLN -> RSZ frozen transfer:   PASSED"
    )
    print(
        "  clean decision check:         PASSED"
    )
    print()
    print(
        "OPERATING-POINT ARTIFACT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()