import csv
import hashlib
import json
from pathlib import Path

from study.processing.processing_common import (
    RawVideoReader,
    probe_video,
)


STUDY_ROOT = (
    Path.home()
    / "deepfake_lab"
    / "study_data"
)

TEMPORAL_PLAN_PATH = (
    STUDY_ROOT
    / "evaluation"
    / "temporal_plan.csv"
)

OUTPUT_PATH = (
    STUDY_ROOT
    / "evaluation"
    / "temporal_extraction_smoke.json"
)

TARGET_FRAME_BUDGET = 32


SELECTORS = (
    {
        "name": "ffpp_validation_original",
        "dataset": "FaceForensics++",
        "role": "validation",
        "subgroup": "original",
    },
    {
        "name": "ffpp_test_original",
        "dataset": "FaceForensics++",
        "role": "test",
        "subgroup": "original",
    },
    {
        "name": "ffpp_test_deepfakes",
        "dataset": "FaceForensics++",
        "role": "test",
        "subgroup": "Deepfakes",
    },
    {
        "name": "celeb_test_real",
        "dataset": "Celeb-DF-v2",
        "role": "test",
        "subgroup": "Celeb-real",
    },
    {
        "name": "celeb_test_synthesis",
        "dataset": "Celeb-DF-v2",
        "role": "test",
        "subgroup": "Celeb-synthesis",
    },
)


def read_plan():
    if not TEMPORAL_PLAN_PATH.is_file():
        raise FileNotFoundError(
            "Temporal plan does not exist: {}".format(
                TEMPORAL_PLAN_PATH
            )
        )

    with TEMPORAL_PLAN_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(
            csv.DictReader(
                file
            )
        )


def sha256_file(path):
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


def row_key(row):
    return (
        row["dataset"],
        row["role"],
        row["subgroup"],
        row["relative_source_path"],
    )


def select_representative_rows(
    plan_rows,
):
    valid_rows = [
        row
        for row in plan_rows
        if (
            row[
                "temporal_plan_status"
            ]
            == "ok"
        )
    ]

    if not valid_rows:
        raise RuntimeError(
            "Temporal plan contains no valid rows."
        )

    selected = {}

    for selector in SELECTORS:
        matches = [
            row
            for row in valid_rows
            if (
                row["dataset"]
                == selector["dataset"]
                and row["role"]
                == selector["role"]
                and row["subgroup"]
                == selector["subgroup"]
            )
        ]

        if not matches:
            raise RuntimeError(
                "No temporal-plan row found for "
                "selector {!r}.".format(
                    selector["name"]
                )
            )

        matches.sort(
            key=lambda row: (
                row[
                    "relative_source_path"
                ]
            )
        )

        row = matches[0]

        selected[
            row_key(
                row
            )
        ] = (
            selector["name"],
            row,
        )

    minimum_row = min(
        valid_rows,
        key=lambda row: (
            int(
                row[
                    "decoded_frame_count"
                ]
            ),
            row_key(
                row
            ),
        ),
    )

    maximum_row = max(
        valid_rows,
        key=lambda row: (
            int(
                row[
                    "decoded_frame_count"
                ]
            ),
            row_key(
                row
            ),
        ),
    )

    minimum_key = row_key(
        minimum_row
    )

    maximum_key = row_key(
        maximum_row
    )

    if minimum_key not in selected:
        selected[
            minimum_key
        ] = (
            "minimum_decoded_frame_count",
            minimum_row,
        )

    if maximum_key not in selected:
        selected[
            maximum_key
        ] = (
            "maximum_decoded_frame_count",
            maximum_row,
        )

    return [
        selected[key]
        for key in sorted(
            selected
        )
    ]


def parse_target_indices(
    row,
):
    try:
        indices = json.loads(
            row[
                "target_indices"
            ]
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Invalid target_indices JSON for {}".format(
                row_key(
                    row
                )
            )
        ) from exc

    if not isinstance(
        indices,
        list,
    ):
        raise RuntimeError(
            "target_indices must be a list for {}".format(
                row_key(
                    row
                )
            )
        )

    if (
        len(
            indices
        )
        != TARGET_FRAME_BUDGET
    ):
        raise RuntimeError(
            "Expected {} frozen indices for {}, got {}.".format(
                TARGET_FRAME_BUDGET,
                row_key(
                    row
                ),
                len(
                    indices
                ),
            )
        )

    if not all(
        isinstance(
            value,
            int,
        )
        for value in indices
    ):
        raise RuntimeError(
            "All target indices must be integers for {}".format(
                row_key(
                    row
                )
            )
        )

    if (
        indices
        != sorted(
            indices
        )
    ):
        raise RuntimeError(
            "Target indices are not sorted for {}".format(
                row_key(
                    row
                )
            )
        )

    if (
        len(
            set(
                indices
            )
        )
        != TARGET_FRAME_BUDGET
    ):
        raise RuntimeError(
            "Target indices contain duplicates for {}".format(
                row_key(
                    row
                )
            )
        )

    return indices


def decode_frozen_positions(
    row,
):
    source_path = (
        Path(
            row[
                "absolute_source_path"
            ]
        )
        .expanduser()
        .resolve()
    )

    if not source_path.is_file():
        raise FileNotFoundError(
            "Source video does not exist: {}".format(
                source_path
            )
        )

    target_indices = (
        parse_target_indices(
            row
        )
    )

    target_set = set(
        target_indices
    )

    expected_decoded_count = int(
        row[
            "decoded_frame_count"
        ]
    )

    if (
        target_indices[-1]
        >= expected_decoded_count
    ):
        raise RuntimeError(
            "Frozen target index is outside expected "
            "decoded sequence for {}".format(
                row_key(
                    row
                )
            )
        )

    probe = probe_video(
        source_path
    )

    width = int(
        probe[
            "width"
        ]
    )

    height = int(
        probe[
            "height"
        ]
    )

    found_indices = []
    frame_hashes = {}

    decoded_index = 0

    with RawVideoReader(
        source_path,
        width=width,
        height=height,
    ) as reader:
        while True:
            frame = reader.read_frame()

            if frame is None:
                break

            if decoded_index in target_set:
                found_indices.append(
                    decoded_index
                )

                digest = hashlib.sha256(
                    frame.tobytes()
                ).hexdigest()

                frame_hashes[
                    str(
                        decoded_index
                    )
                ] = digest

            decoded_index += 1

    if (
        decoded_index
        != expected_decoded_count
    ):
        raise RuntimeError(
            "Runtime decoded-frame count mismatch for {}: "
            "plan={}, runtime={}.".format(
                row_key(
                    row
                ),
                expected_decoded_count,
                decoded_index,
            )
        )

    if (
        found_indices
        != target_indices
    ):
        missing = [
            index
            for index in target_indices
            if index not in found_indices
        ]

        unexpected = [
            index
            for index in found_indices
            if index not in target_set
        ]

        raise RuntimeError(
            "Frozen temporal extraction mismatch for {}.\n"
            "Expected: {}\n"
            "Found:    {}\n"
            "Missing:  {}\n"
            "Extra:    {}".format(
                row_key(
                    row
                ),
                target_indices,
                found_indices,
                missing,
                unexpected,
            )
        )

    if (
        len(
            frame_hashes
        )
        != TARGET_FRAME_BUDGET
    ):
        raise RuntimeError(
            "Expected {} extracted-frame hashes for {}, "
            "got {}.".format(
                TARGET_FRAME_BUDGET,
                row_key(
                    row
                ),
                len(
                    frame_hashes
                ),
            )
        )

    return {
        "dataset": row[
            "dataset"
        ],
        "role": row[
            "role"
        ],
        "subgroup": row[
            "subgroup"
        ],
        "base_video_id": row[
            "base_video_id"
        ],
        "relative_source_path": row[
            "relative_source_path"
        ],
        "absolute_source_path": str(
            source_path
        ),
        "expected_decoded_frame_count": (
            expected_decoded_count
        ),
        "runtime_decoded_frame_count": (
            decoded_index
        ),
        "target_frame_budget": (
            TARGET_FRAME_BUDGET
        ),
        "target_indices": (
            target_indices
        ),
        "first_target_index": (
            target_indices[0]
        ),
        "last_target_index": (
            target_indices[-1]
        ),
        "extracted_frame_count": len(
            found_indices
        ),
        "frame_sha256": (
            frame_hashes
        ),
        "status": "ok",
    }


def main():
    plan_rows = read_plan()

    selected_rows = (
        select_representative_rows(
            plan_rows
        )
    )

    results = []

    print(
        "TEMPORAL EXTRACTION RUNTIME SMOKE"
    )
    print(
        "  temporal-plan SHA-256:     {}".format(
            sha256_file(
                TEMPORAL_PLAN_PATH
            )
        )
    )
    print(
        "  selected videos:           {}".format(
            len(
                selected_rows
            )
        )
    )
    print()

    for (
        position,
        (
            selection_reason,
            row,
        ),
    ) in enumerate(
        selected_rows,
        start=1,
    ):
        result = decode_frozen_positions(
            row
        )

        result[
            "selection_reason"
        ] = selection_reason

        results.append(
            result
        )

        print(
            "  [{}/{}] {} | {} | {} | {}".format(
                position,
                len(
                    selected_rows
                ),
                result[
                    "dataset"
                ],
                result[
                    "role"
                ],
                result[
                    "subgroup"
                ],
                result[
                    "base_video_id"
                ],
            )
        )

        print(
            "        decoded: {} / {}".format(
                result[
                    "runtime_decoded_frame_count"
                ],
                result[
                    "expected_decoded_frame_count"
                ],
            )
        )

        print(
            "        frozen targets: {} / {}".format(
                result[
                    "extracted_frame_count"
                ],
                result[
                    "target_frame_budget"
                ],
            )
        )

        print(
            "        first/last: {} / {}".format(
                result[
                    "first_target_index"
                ],
                result[
                    "last_target_index"
                ],
            )
        )

    output = {
        "schema_version": 1,
        "temporal_plan_path": str(
            TEMPORAL_PLAN_PATH
        ),
        "temporal_plan_sha256": (
            sha256_file(
                TEMPORAL_PLAN_PATH
            )
        ),
        "selection_policy": (
            "deterministic representative categories plus "
            "global minimum/maximum decoded-frame-count records"
        ),
        "selected_video_count": len(
            results
        ),
        "target_frame_budget": (
            TARGET_FRAME_BUDGET
        ),
        "all_status_ok": all(
            result[
                "status"
            ]
            == "ok"
            for result in results
        ),
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write(
            "\n"
        )

    print()
    print(
        "  smoke artifact:            {}".format(
            OUTPUT_PATH
        )
    )

    print()
    print(
        "TEMPORAL EXTRACTION RUNTIME SMOKE PASSED"
    )


if __name__ == "__main__":
    main()