import csv
import hashlib
import json
from collections import Counter
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "study"
    / "materialization"
    / "artifacts"
)

VIDEO_PLAN_PATH = (
    ARTIFACT_DIR
    / "video_frame_plan.csv"
)

FRAME_PLAN_PATH = (
    ARTIFACT_DIR
    / "frame_target_plan.csv"
)

SUMMARY_PATH = (
    ARTIFACT_DIR
    / "frame_plan_summary.json"
)

TARGET_FRAME_BUDGET = 32


def sha256_file(
    path,
):
    digest = hashlib.sha256()

    with Path(path).open(
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


def read_csv(
    path,
):
    with Path(path).open(
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
    video_rows = read_csv(
        VIDEO_PLAN_PATH
    )

    frame_rows = read_csv(
        FRAME_PLAN_PATH
    )

    with SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(
            file
        )

    assert len(
        video_rows
    ) == 3600

    assert len(
        {
            row[
                "relative_source_path"
            ]
            for row in video_rows
        }
    ) == 3600

    assert Counter(
        row[
            "execution_split"
        ]
        for row in video_rows
    ) == {
        "train": 2880,
        "test": 720,
    }

    assert Counter(
        row[
            "role"
        ]
        for row in video_rows
    ) == {
        "fit": 2880,
        "development": 720,
    }

    assert Counter(
        row[
            "study_label"
        ]
        for row in video_rows
    ) == {
        "0": 720,
        "1": 2880,
    }

    assert Counter(
        (
            row[
                "manipulation"
            ]
            or "REAL"
        )
        for row in video_rows
    ) == {
        "REAL": 720,
        "DeepFakes": 720,
        "Face2Face": 720,
        "FaceSwap": 720,
        "NeuralTextures": 720,
    }

    status_counts = Counter(
        row[
            "status"
        ]
        for row in video_rows
    )

    assert status_counts == {
        "planned": 3600
    }

    assert len(
        frame_rows
    ) == (
        3600
        * TARGET_FRAME_BUDGET
    )

    video_by_source = {
        row[
            "relative_source_path"
        ]: row
        for row in video_rows
    }

    frames_by_source = defaultdict(
        list
    )

    output_paths = set()

    for frame_row in frame_rows:
        source_path = frame_row[
            "relative_source_path"
        ]

        assert source_path in (
            video_by_source
        )

        frames_by_source[
            source_path
        ].append(
            frame_row
        )

        output_path = frame_row[
            "output_relative_path"
        ]

        if output_path in (
            output_paths
        ):
            raise AssertionError(
                "Duplicate output frame path: "
                f"{output_path}"
            )

        output_paths.add(
            output_path
        )

    assert len(
        frames_by_source
    ) == 3600

    for (
        source_path,
        rows,
    ) in frames_by_source.items():
        assert len(
            rows
        ) == (
            TARGET_FRAME_BUDGET
        )

        positions = [
            int(
                row[
                    "temporal_position"
                ]
            )
            for row in rows
        ]

        indices = [
            int(
                row[
                    "source_frame_index"
                ]
            )
            for row in rows
        ]

        assert positions == list(
            range(
                TARGET_FRAME_BUDGET
            )
        )

        assert indices == sorted(
            indices
        )

        assert len(
            set(
                indices
            )
        ) == (
            TARGET_FRAME_BUDGET
        )

        reported_frame_count = int(
            video_by_source[
                source_path
            ][
                "reported_frame_count"
            ]
        )

        assert indices[
            0
        ] >= 0

        assert indices[
            -1
        ] < reported_frame_count

        stored_indices = json.loads(
            video_by_source[
                source_path
            ][
                "target_indices"
            ]
        )

        assert stored_indices == indices

    assert summary[
        "sampling"
    ][
        "method"
    ] == (
        "equal_bin_midpoint_v1"
    )

    assert summary[
        "sampling"
    ][
        "target_frame_budget"
    ] == 32

    assert summary[
        "counts"
    ][
        "aggregate_video_rows"
    ] == 3600

    assert summary[
        "counts"
    ][
        "planned_video_rows"
    ] == 3600

    assert summary[
        "counts"
    ][
        "frame_target_rows"
    ] == 115200

    assert (
        summary[
            "outputs"
        ][
            "video_frame_plan.csv"
        ][
            "sha256"
        ]
        == sha256_file(
            VIDEO_PLAN_PATH
        )
    )

    assert (
        summary[
            "outputs"
        ][
            "frame_target_plan.csv"
        ][
            "sha256"
        ]
        == sha256_file(
            FRAME_PLAN_PATH
        )
    )

    reported_counts = [
        int(
            row[
                "reported_frame_count"
            ]
        )
        for row in video_rows
    ]

    print(
        "FRAME PLAN VALIDATION"
    )

    print(
        "  videos:                   3600"
    )

    print(
        "  FIT videos:               2880"
    )

    print(
        "  DEV videos:               720"
    )

    print(
        "  target frames/video:      32"
    )

    print(
        "  total target frames:      115200"
    )

    print(
        "  unique output paths:      115200"
    )

    print(
        "  min reported frames:      "
        f"{min(reported_counts)}"
    )

    print(
        "  max reported frames:      "
        f"{max(reported_counts)}"
    )

    print(
        "  duplicate target indices: none"
    )

    print(
        "  sampling method:          "
        "equal_bin_midpoint_v1"
    )

    print()
    print(
        "FRAME PLAN VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()