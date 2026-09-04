from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path


def parse_rate(value):
    if not value:
        return None

    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None

    if rate <= 0:
        return None

    return rate


def rates_equal(first, second, tolerance=1e-9):
    first_rate = parse_rate(first)
    second_rate = parse_rate(second)

    if first_rate is None or second_rate is None:
        return False

    return abs(float(first_rate) - float(second_rate)) <= tolerance


def probe_video(path, ffprobe_bin):
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,pix_fmt,width,height,"
            "r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames:"
            "format=duration"
        ),
        "-of",
        "json",
        str(path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return {
            "status": "ffprobe_error",
            "error": result.stderr.strip(),
        }

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "status": "json_error",
            "error": str(exc),
        }

    streams = data.get("streams", [])

    if not streams:
        return {
            "status": "no_video_stream",
            "error": "No video stream returned by ffprobe.",
        }

    stream = streams[0]
    format_info = data.get("format", {})

    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "invalid_geometry",
            "error": "Unable to determine video width/height.",
        }

    nb_frames = stream.get("nb_frames")
    nb_read_frames = stream.get("nb_read_frames")

    try:
        metadata_frame_count = int(nb_frames) if nb_frames not in (None, "N/A") else None
    except ValueError:
        metadata_frame_count = None

    try:
        counted_frame_count = (
            int(nb_read_frames)
            if nb_read_frames not in (None, "N/A")
            else None
        )
    except ValueError:
        counted_frame_count = None

    r_frame_rate = stream.get("r_frame_rate")
    avg_frame_rate = stream.get("avg_frame_rate")

    return {
        "status": "ok",
        "error": "",
        "codec_name": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "width": width,
        "height": height,
        "r_frame_rate": r_frame_rate,
        "avg_frame_rate": avg_frame_rate,
        "nominal_rate_match": rates_equal(
            r_frame_rate,
            avg_frame_rate,
        ),
        "metadata_frame_count": metadata_frame_count,
        "counted_frame_count": counted_frame_count,
        "frame_count_match": (
            metadata_frame_count == counted_frame_count
            if metadata_frame_count is not None
            and counted_frame_count is not None
            else None
        ),
        "has_32_frames": (
            counted_frame_count >= 32
            if counted_frame_count is not None
            else None
        ),
        "even_width": width % 2 == 0,
        "even_height": height % 2 == 0,
        "duration": format_info.get("duration"),
    }


def discover_videos(dataset_root, include_subdirs):
    videos = []

    if include_subdirs:
        roots = []

        for subdir in include_subdirs:
            root = dataset_root / subdir

            if not root.is_dir():
                raise FileNotFoundError(
                    f"Included subdirectory does not exist: {root}"
                )

            roots.append((subdir, root))
    else:
        roots = [(".", dataset_root)]

    for subgroup, root in roots:
        for path in sorted(root.rglob("*.mp4")):
            if path.is_file():
                videos.append((subgroup, path))

    return videos


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
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

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_summary(rows):
    total = len(rows)
    ok_rows = [row for row in rows if row["status"] == "ok"]

    status_counts = Counter(row["status"] for row in rows)
    codec_counts = Counter(
        row["codec_name"]
        for row in ok_rows
        if row.get("codec_name")
    )
    pixel_format_counts = Counter(
        row["pixel_format"]
        for row in ok_rows
        if row.get("pixel_format")
    )
    resolution_counts = Counter(
        "{}x{}".format(row["width"], row["height"])
        for row in ok_rows
    )
    fps_counts = Counter(
        row["avg_frame_rate"]
        for row in ok_rows
        if row.get("avg_frame_rate")
    )
    subgroup_counts = Counter(row["subgroup"] for row in rows)

    return {
        "total_videos": total,
        "successful_probes": len(ok_rows),
        "failed_probes": total - len(ok_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "subgroup_counts": dict(sorted(subgroup_counts.items())),
        "codec_counts": dict(sorted(codec_counts.items())),
        "pixel_format_counts": dict(sorted(pixel_format_counts.items())),
        "resolution_counts": dict(sorted(resolution_counts.items())),
        "avg_frame_rate_counts": dict(sorted(fps_counts.items())),
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
            if not row["even_width"] or not row["even_height"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Preflight source videos before generating study "
            "processing conditions."
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
        "--include-subdir",
        action="append",
        default=[],
        help=(
            "Relative dataset subdirectory to include. "
            "May be specified multiple times."
        ),
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
            f"Dataset root does not exist: {args.dataset_root}"
        )

    videos = discover_videos(
        dataset_root=args.dataset_root,
        include_subdirs=args.include_subdir,
    )

    if not videos:
        raise RuntimeError(
            f"No MP4 videos found under {args.dataset_root}"
        )

    print("Dataset: {}".format(args.dataset_name))
    print("Root:    {}".format(args.dataset_root))
    print("Videos:  {}".format(len(videos)))
    print()

    rows = []

    for index, (subgroup, path) in enumerate(videos, start=1):
        relative_path = path.relative_to(args.dataset_root)

        result = probe_video(
            path=path,
            ffprobe_bin=args.ffprobe,
        )

        row = {
            "dataset": args.dataset_name,
            "subgroup": subgroup,
            "relative_path": str(relative_path),
            "absolute_path": str(path),
        }
        row.update(result)

        rows.append(row)

        if index % 100 == 0 or index == len(videos):
            print(
                "Probed {}/{} videos".format(
                    index,
                    len(videos),
                )
            )

    write_csv(
        rows=rows,
        output_path=args.output_csv,
    )

    summary = make_summary(rows)

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
    print("Total videos:              {}".format(summary["total_videos"]))
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