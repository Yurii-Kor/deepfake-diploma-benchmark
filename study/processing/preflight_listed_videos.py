from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from preflight_source_videos import probe_video


def read_video_list(list_path):
    records = []

    with list_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            parts = stripped.split(maxsplit=1)

            if len(parts) != 2:
                raise ValueError(
                    "Invalid list entry at line {}: {!r}".format(
                        line_number,
                        stripped,
                    )
                )

            source_label, relative_path = parts

            records.append(
                {
                    "line_number": line_number,
                    "source_label": source_label,
                    "relative_path": relative_path,
                }
            )

    if not records:
        raise RuntimeError(
            "No video records found in: {}".format(list_path)
        )

    return records


def validate_list(records, dataset_root):
    seen_paths = set()
    duplicate_paths = []
    missing_paths = []

    for record in records:
        relative_path = record["relative_path"]

        if relative_path in seen_paths:
            duplicate_paths.append(relative_path)
        else:
            seen_paths.add(relative_path)

        absolute_path = dataset_root / relative_path

        if not absolute_path.is_file():
            missing_paths.append(relative_path)

    return duplicate_paths, missing_paths


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "list_line_number",
        "source_label",
        "subgroup",
        "relative_path",
        "absolute_path",
        "status",
        "error",
        "codec_name",
        "pixel_format",
        "width",
        "height",
        "r_frame_rate",
        "avg_frame_rate",
        "nominal_rate_match",
        "metadata_frame_count",
        "counted_frame_count",
        "frame_count_match",
        "has_32_frames",
        "even_width",
        "even_height",
        "duration",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def make_summary(rows, duplicate_paths, missing_paths):
    ok_rows = [
        row
        for row in rows
        if row["status"] == "ok"
    ]

    return {
        "total_list_records": len(rows),
        "successful_probes": len(ok_rows),
        "failed_probes": len(rows) - len(ok_rows),
        "duplicate_list_paths": len(duplicate_paths),
        "missing_list_paths": len(missing_paths),
        "source_label_counts": dict(
            sorted(
                Counter(
                    row["source_label"]
                    for row in rows
                ).items()
            )
        ),
        "subgroup_counts": dict(
            sorted(
                Counter(
                    row["subgroup"]
                    for row in rows
                ).items()
            )
        ),
        "status_counts": dict(
            sorted(
                Counter(
                    row["status"]
                    for row in rows
                ).items()
            )
        ),
        "codec_counts": dict(
            sorted(
                Counter(
                    row["codec_name"]
                    for row in ok_rows
                    if row.get("codec_name")
                ).items()
            )
        ),
        "pixel_format_counts": dict(
            sorted(
                Counter(
                    row["pixel_format"]
                    for row in ok_rows
                    if row.get("pixel_format")
                ).items()
            )
        ),
        "resolution_counts": dict(
            sorted(
                Counter(
                    "{}x{}".format(
                        row["width"],
                        row["height"],
                    )
                    for row in ok_rows
                ).items()
            )
        ),
        "avg_frame_rate_counts": dict(
            sorted(
                Counter(
                    row["avg_frame_rate"]
                    for row in ok_rows
                    if row.get("avg_frame_rate")
                ).items()
            )
        ),
        "nominal_rate_mismatches": sum(
            1
            for row in ok_rows
            if row["nominal_rate_match"] is False
        ),
        "metadata_count_mismatches": sum(
            1
            for row in ok_rows
            if row["frame_count_match"] is False
        ),
        "videos_below_32_frames": sum(
            1
            for row in ok_rows
            if row["has_32_frames"] is False
        ),
        "odd_dimension_videos": sum(
            1
            for row in ok_rows
            if not row["even_width"]
            or not row["even_height"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Preflight videos referenced by an authoritative "
            "dataset list file."
        )
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--list-file",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-summary",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
    )
    args = parser.parse_args()

    if not args.dataset_root.is_dir():
        raise FileNotFoundError(
            "Dataset root does not exist: {}".format(
                args.dataset_root
            )
        )

    if not args.list_file.is_file():
        raise FileNotFoundError(
            "List file does not exist: {}".format(
                args.list_file
            )
        )

    records = read_video_list(args.list_file)

    duplicate_paths, missing_paths = validate_list(
        records=records,
        dataset_root=args.dataset_root,
    )

    print("Dataset:       {}".format(args.dataset_name))
    print("Root:          {}".format(args.dataset_root))
    print("List:          {}".format(args.list_file))
    print("List records:  {}".format(len(records)))
    print("Duplicates:    {}".format(len(duplicate_paths)))
    print("Missing:       {}".format(len(missing_paths)))
    print()

    if duplicate_paths:
        raise RuntimeError(
            "The authoritative list contains duplicate paths."
        )

    if missing_paths:
        print("Missing paths:")

        for path in missing_paths:
            print("  {}".format(path))

        raise RuntimeError(
            "{} listed videos are missing.".format(
                len(missing_paths)
            )
        )

    rows = []

    for index, record in enumerate(records, start=1):
        relative_path = record["relative_path"]
        absolute_path = args.dataset_root / relative_path

        subgroup = Path(relative_path).parts[0]

        result = probe_video(
            path=absolute_path,
            ffprobe_bin=args.ffprobe,
        )

        row = {
            "dataset": args.dataset_name,
            "list_line_number": record["line_number"],
            "source_label": record["source_label"],
            "subgroup": subgroup,
            "relative_path": relative_path,
            "absolute_path": str(absolute_path),
        }

        row.update(result)
        rows.append(row)

        if (
            index % 50 == 0
            or index == len(records)
        ):
            print(
                "Probed {}/{} videos".format(
                    index,
                    len(records),
                )
            )

    write_csv(
        rows=rows,
        output_path=args.output_csv,
    )

    summary = make_summary(
        rows=rows,
        duplicate_paths=duplicate_paths,
        missing_paths=missing_paths,
    )

    args.output_summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output_summary.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    print()
    print("PREFLIGHT COMPLETED")
    print(
        "List records:              {}".format(
            summary["total_list_records"]
        )
    )
    print(
        "Successful probes:         {}".format(
            summary["successful_probes"]
        )
    )
    print(
        "Failed probes:             {}".format(
            summary["failed_probes"]
        )
    )
    print(
        "Duplicate paths:           {}".format(
            summary["duplicate_list_paths"]
        )
    )
    print(
        "Missing paths:             {}".format(
            summary["missing_list_paths"]
        )
    )
    print(
        "Nominal-rate mismatches:   {}".format(
            summary["nominal_rate_mismatches"]
        )
    )
    print(
        "Frame-count mismatches:    {}".format(
            summary["metadata_count_mismatches"]
        )
    )
    print(
        "Videos below 32 frames:    {}".format(
            summary["videos_below_32_frames"]
        )
    )
    print(
        "Odd-dimension videos:      {}".format(
            summary["odd_dimension_videos"]
        )
    )
    print()
    print("CSV:     {}".format(args.output_csv))
    print("Summary: {}".format(args.output_summary))


if __name__ == "__main__":
    main()