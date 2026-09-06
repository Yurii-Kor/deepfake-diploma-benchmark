import csv
import json
from collections import Counter
from pathlib import Path

import cv2


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

VIDEO_PLAN_PATH = (
    PROJECT_ROOT
    / "study"
    / "materialization"
    / "artifacts"
    / "video_frame_plan.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "study"
    / "materialization"
    / "artifacts"
    / "decoded_frame_count_audit.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "study"
    / "materialization"
    / "artifacts"
    / "decoded_frame_count_summary.json"
)


FIELDS = [
    "execution_split",
    "role",
    "relative_source_path",
    "reported_frame_count",
    "decoded_frame_count",
    "count_difference",
    "decode_status",
    "failure_reason",
]


def decode_frame_count(
    source_path,
):
    capture = cv2.VideoCapture(
        str(source_path)
    )

    if not capture.isOpened():
        return (
            0,
            "open_failed",
            "cv2.VideoCapture could not open source",
        )

    decoded_count = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            if frame is None:
                return (
                    decoded_count,
                    "decode_failed",
                    (
                        "VideoCapture returned success "
                        "with frame=None"
                    ),
                )

            decoded_count += 1

    finally:
        capture.release()

    if decoded_count == 0:
        return (
            0,
            "decode_failed",
            "no frames decoded",
        )

    return (
        decoded_count,
        "ok",
        "",
    )


def main():
    with VIDEO_PLAN_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        source_rows = list(
            csv.DictReader(file)
        )

    results = []

    total = len(source_rows)

    print(
        "DECODED FRAME-COUNT PREFLIGHT"
    )
    print(
        f"  videos:                    {total}"
    )
    print()

    for index, row in enumerate(
        source_rows,
        start=1,
    ):
        source_path = Path(
            row["absolute_source_path"]
        )

        reported_count = int(
            row["reported_frame_count"]
        )

        (
            decoded_count,
            decode_status,
            failure_reason,
        ) = decode_frame_count(
            source_path
        )

        results.append(
            {
                "execution_split": (
                    row["execution_split"]
                ),
                "role": row["role"],
                "relative_source_path": (
                    row["relative_source_path"]
                ),
                "reported_frame_count": (
                    reported_count
                ),
                "decoded_frame_count": (
                    decoded_count
                ),
                "count_difference": (
                    decoded_count
                    - reported_count
                ),
                "decode_status": (
                    decode_status
                ),
                "failure_reason": (
                    failure_reason
                ),
            }
        )

        if (
            index % 100 == 0
            or index == total
        ):
            mismatches = sum(
                int(
                    item[
                        "count_difference"
                    ]
                )
                != 0
                for item in results
            )

            failures = sum(
                item[
                    "decode_status"
                ]
                != "ok"
                for item in results
            )

            print(
                f"  decoded:                   "
                f"{index}/{total}  "
                f"mismatches={mismatches}  "
                f"failures={failures}"
            )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
        )

        writer.writeheader()
        writer.writerows(results)

    mismatched = [
        row
        for row in results
        if int(
            row[
                "count_difference"
            ]
        )
        != 0
    ]

    failed = [
        row
        for row in results
        if row[
            "decode_status"
        ]
        != "ok"
    ]

    insufficient = [
        row
        for row in results
        if int(
            row[
                "decoded_frame_count"
            ]
        )
        < 32
    ]

    decoded_counts = [
        int(
            row[
                "decoded_frame_count"
            ]
        )
        for row in results
        if row[
            "decode_status"
        ]
        == "ok"
    ]

    summary = {
        "video_count": total,
        "decode_status": dict(
            Counter(
                row[
                    "decode_status"
                ]
                for row in results
            )
        ),
        "reported_vs_decoded_mismatch_count": (
            len(mismatched)
        ),
        "insufficient_decoded_frames_count": (
            len(insufficient)
        ),
        "minimum_decoded_frame_count": (
            min(decoded_counts)
            if decoded_counts
            else None
        ),
        "maximum_decoded_frame_count": (
            max(decoded_counts)
            if decoded_counts
            else None
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write("\n")

    print()
    print(
        "DECODE PREFLIGHT RESULT"
    )
    print(
        f"  successfully decoded:      "
        f"{total - len(failed)}"
    )
    print(
        f"  decode failures:           "
        f"{len(failed)}"
    )
    print(
        f"  count mismatches:          "
        f"{len(mismatched)}"
    )
    print(
        f"  decoded videos <32 frames: "
        f"{len(insufficient)}"
    )

    if decoded_counts:
        print(
            f"  min decoded frames:        "
            f"{min(decoded_counts)}"
        )
        print(
            f"  max decoded frames:        "
            f"{max(decoded_counts)}"
        )

    print()
    print(
        f"  audit CSV:                 "
        f"{OUTPUT_PATH}"
    )

    print(
        f"  summary:                   "
        f"{SUMMARY_PATH}"
    )

    if failed:
        raise RuntimeError(
            "One or more source videos failed "
            "sequential decoding."
        )

    if insufficient:
        raise RuntimeError(
            "One or more videos contain fewer than "
            "32 decoded frames."
        )

    print()
    print(
        "DECODED FRAME-COUNT PREFLIGHT PASSED"
    )


if __name__ == "__main__":
    main()