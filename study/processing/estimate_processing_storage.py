from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


BYTES_PER_BGR_PIXEL = 3
MIB = 1024 ** 2
GIB = 1024 ** 3
TIB = 1024 ** 4

# ---------------------------------------------------------------------
# Representative pilot calibration
# ---------------------------------------------------------------------
#
# These factors are used only for disk-capacity planning.
# FFV1 compression is content-dependent, so the resulting values are
# estimates rather than guarantees.
#
# FF++ representative:
#   source metadata: 640 x 480, 396 decoded frames
#   RSZ smoke output: ~71 MiB
#   BLR smoke output: ~60 MiB
#
# Celeb-DF-v2 representative:
#   source metadata: 856 x 478, 453 decoded frames
#   RSZ smoke output: ~98 MiB
#   BLR smoke output: ~77 MiB
#
# The du -h values are rounded filesystem-size observations, so these
# ratios intentionally remain planning approximations.

FFPP_SAMPLE_RAW_BYTES = (
    640
    * 480
    * 396
    * BYTES_PER_BGR_PIXEL
)

CELEB_SAMPLE_RAW_BYTES = (
    856
    * 478
    * 453
    * BYTES_PER_BGR_PIXEL
)

DEFAULT_FFPP_RSZ_RATIO = (
    71 * MIB
    / FFPP_SAMPLE_RAW_BYTES
)

DEFAULT_FFPP_BLR_RATIO = (
    60 * MIB
    / FFPP_SAMPLE_RAW_BYTES
)

DEFAULT_CELEB_RSZ_RATIO = (
    98 * MIB
    / CELEB_SAMPLE_RAW_BYTES
)

DEFAULT_CELEB_BLR_RATIO = (
    77 * MIB
    / CELEB_SAMPLE_RAW_BYTES
)


def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def normalize_dataset_name(
    dataset,
):
    """
    Normalize historical preflight dataset names to the frozen
    processing-manifest namespace.

    Historical FF++ preflight files are inconsistent:

        FaceForensics++/original
        FaceForensics++/Deepfakes
        FaceForensics++/Face2Face
        FaceForensics++
        ...

    The frozen processing manifest deliberately uses:

        dataset = FaceForensics++
        subgroup = original / Deepfakes / ...

    Celeb-DF-v2 already uses a stable dataset name.
    """

    dataset = dataset.strip()

    if (
        dataset == "FaceForensics++"
        or dataset.startswith(
            "FaceForensics++/"
        )
    ):
        return "FaceForensics++"

    return dataset


def load_preflight_rows(
    ffpp_preflight_dir,
    celeb_preflight_csv,
):
    rows = {}

    ffpp_subgroups = (
        "original",
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
    )

    for subgroup in ffpp_subgroups:
        path = (
            ffpp_preflight_dir
            / "{}_preflight.csv".format(
                subgroup
            )
        )

        if not path.is_file():
            raise FileNotFoundError(
                "Missing FF++ preflight CSV: "
                "{}".format(path)
            )

        for row in read_csv(path):
            dataset = normalize_dataset_name(
                row["dataset"]
            )

            relative_path = (
                row["relative_path"].strip()
            )

            key = (
                dataset,
                relative_path,
            )

            if key in rows:
                raise ValueError(
                    "Duplicate normalized preflight key: "
                    "{}".format(key)
                )

            rows[key] = row

    if not celeb_preflight_csv.is_file():
        raise FileNotFoundError(
            "Missing Celeb preflight CSV: "
            "{}".format(
                celeb_preflight_csv
            )
        )

    for row in read_csv(
        celeb_preflight_csv
    ):
        dataset = normalize_dataset_name(
            row["dataset"]
        )

        relative_path = (
            row["relative_path"].strip()
        )

        key = (
            dataset,
            relative_path,
        )

        if key in rows:
            raise ValueError(
                "Duplicate normalized preflight key: "
                "{}".format(key)
            )

        rows[key] = row

    return rows


def bytes_to_gib(
    value,
):
    return (
        value
        / GIB
    )


def bytes_to_tib(
    value,
):
    return (
        value
        / TIB
    )


def get_condition_ratios(
    dataset,
    args,
):
    if dataset == "FaceForensics++":
        return (
            args.ffpp_rsz_ratio,
            args.ffpp_blr_ratio,
        )

    if dataset == "Celeb-DF-v2":
        return (
            args.celeb_rsz_ratio,
            args.celeb_blr_ratio,
        )

    raise ValueError(
        "No storage calibration "
        "available for dataset: "
        "{}".format(dataset)
    )


def update_group(
    target,
    frame_count,
    raw_bgr_bytes,
    rsz_ratio,
    blr_ratio,
):
    target["videos"] += 1

    target[
        "decoded_frames"
    ] += frame_count

    target[
        "raw_bgr_bytes"
    ] += raw_bgr_bytes

    target[
        "rsz_estimated_bytes"
    ] += int(
        raw_bgr_bytes
        * rsz_ratio
    )

    target[
        "blr_estimated_bytes"
    ] += int(
        raw_bgr_bytes
        * blr_ratio
    )


def make_group_summary(
    values,
):
    lossless_total = (
        values[
            "rsz_estimated_bytes"
        ]
        + values[
            "blr_estimated_bytes"
        ]
    )

    headroom_25 = (
        lossless_total
        * 1.25
    )

    return {
        "videos": (
            values["videos"]
        ),
        "decoded_frames": (
            values[
                "decoded_frames"
            ]
        ),
        "raw_bgr_gib": round(
            bytes_to_gib(
                values[
                    "raw_bgr_bytes"
                ]
            ),
            3,
        ),
        "rsz_estimated_gib": round(
            bytes_to_gib(
                values[
                    "rsz_estimated_bytes"
                ]
            ),
            3,
        ),
        "blr_estimated_gib": round(
            bytes_to_gib(
                values[
                    "blr_estimated_bytes"
                ]
            ),
            3,
        ),
        "rsz_plus_blr_estimated_gib": round(
            bytes_to_gib(
                lossless_total
            ),
            3,
        ),
        "rsz_plus_blr_estimated_tib": round(
            bytes_to_tib(
                lossless_total
            ),
            4,
        ),
        "rsz_plus_blr_with_25pct_headroom_gib": round(
            bytes_to_gib(
                headroom_25
            ),
            3,
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate storage requirements "
            "for the frozen processing manifest "
            "using existing source-preflight metadata."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--ffpp-preflight-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--celeb-preflight-csv",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--ffpp-rsz-ratio",
        type=float,
        default=DEFAULT_FFPP_RSZ_RATIO,
    )

    parser.add_argument(
        "--ffpp-blr-ratio",
        type=float,
        default=DEFAULT_FFPP_BLR_RATIO,
    )

    parser.add_argument(
        "--celeb-rsz-ratio",
        type=float,
        default=DEFAULT_CELEB_RSZ_RATIO,
    )

    parser.add_argument(
        "--celeb-blr-ratio",
        type=float,
        default=DEFAULT_CELEB_BLR_RATIO,
    )

    args = parser.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(
            "Processing manifest "
            "does not exist: "
            "{}".format(
                args.manifest
            )
        )

    manifest_rows = read_csv(
        args.manifest
    )

    preflight_rows = (
        load_preflight_rows(
            ffpp_preflight_dir=(
                args.ffpp_preflight_dir
            ),
            celeb_preflight_csv=(
                args.celeb_preflight_csv
            ),
        )
    )

    totals = defaultdict(
        lambda: {
            "videos": 0,
            "decoded_frames": 0,
            "raw_bgr_bytes": 0,
            "rsz_estimated_bytes": 0,
            "blr_estimated_bytes": 0,
        }
    )

    missing_preflight = []

    for record in manifest_rows:
        dataset = normalize_dataset_name(
            record["dataset"]
        )

        relative_path = (
            record[
                "relative_source_path"
            ].strip()
        )

        key = (
            dataset,
            relative_path,
        )

        preflight = (
            preflight_rows.get(
                key
            )
        )

        if preflight is None:
            missing_preflight.append(
                {
                    "dataset": dataset,
                    "role": (
                        record["role"]
                    ),
                    "subgroup": (
                        record["subgroup"]
                    ),
                    "relative_source_path": (
                        relative_path
                    ),
                }
            )

            continue

        status = (
            preflight[
                "status"
            ].strip()
        )

        if status != "ok":
            raise RuntimeError(
                "Manifest video has "
                "non-OK preflight status: "
                "{} | {} | {}".format(
                    dataset,
                    relative_path,
                    status,
                )
            )

        width = int(
            preflight["width"]
        )

        height = int(
            preflight["height"]
        )

        frame_count = int(
            preflight[
                "counted_frame_count"
            ]
        )

        if (
            width <= 0
            or height <= 0
            or frame_count <= 0
        ):
            raise RuntimeError(
                "Invalid preflight geometry "
                "or frame count: "
                "{} | {} | "
                "{}x{} | frames={}".format(
                    dataset,
                    relative_path,
                    width,
                    height,
                    frame_count,
                )
            )

        raw_bgr_bytes = (
            width
            * height
            * BYTES_PER_BGR_PIXEL
            * frame_count
        )

        (
            rsz_ratio,
            blr_ratio,
        ) = get_condition_ratios(
            dataset,
            args,
        )

        dataset_role_key = (
            "{}|{}".format(
                dataset,
                record["role"],
            )
        )

        dataset_role_subgroup_key = (
            "{}|{}|{}".format(
                dataset,
                record["role"],
                record["subgroup"],
            )
        )

        for target_key in (
            "ALL",
            dataset,
            dataset_role_key,
            dataset_role_subgroup_key,
        ):
            update_group(
                target=totals[
                    target_key
                ],
                frame_count=frame_count,
                raw_bgr_bytes=raw_bgr_bytes,
                rsz_ratio=rsz_ratio,
                blr_ratio=blr_ratio,
            )

    if missing_preflight:
        print(
            "Missing preflight metadata: "
            "{}".format(
                len(
                    missing_preflight
                )
            )
        )

        for item in (
            missing_preflight[:30]
        ):
            print(
                "{} | {} | {} | {}".format(
                    item["dataset"],
                    item["role"],
                    item["subgroup"],
                    item[
                        "relative_source_path"
                    ],
                )
            )

        if len(
            missing_preflight
        ) > 30:
            print(
                "... {} additional "
                "missing records".format(
                    len(
                        missing_preflight
                    )
                    - 30
                )
            )

        raise RuntimeError(
            "Storage estimate cannot "
            "be completed."
        )

    matched = (
        totals["ALL"]["videos"]
    )

    if matched != len(
        manifest_rows
    ):
        raise RuntimeError(
            "Preflight join cardinality "
            "mismatch: manifest={}, "
            "matched={}".format(
                len(
                    manifest_rows
                ),
                matched,
            )
        )

    summary = {
        "manifest_videos": (
            len(
                manifest_rows
            )
        ),
        "matched_preflight_videos": (
            matched
        ),
        "estimation_parameters": {
            "bytes_per_bgr_pixel": (
                BYTES_PER_BGR_PIXEL
            ),
            "ffpp_rsz_ffv1_fraction_of_raw_bgr": (
                args.ffpp_rsz_ratio
            ),
            "ffpp_blr_ffv1_fraction_of_raw_bgr": (
                args.ffpp_blr_ratio
            ),
            "celeb_rsz_ffv1_fraction_of_raw_bgr": (
                args.celeb_rsz_ratio
            ),
            "celeb_blr_ffv1_fraction_of_raw_bgr": (
                args.celeb_blr_ratio
            ),
            "pilot_basis": {
                "ffpp": {
                    "dimensions": (
                        "640x480"
                    ),
                    "frames": 396,
                    "rsz_observed_mib": 71,
                    "blr_observed_mib": 60,
                },
                "celeb_df_v2": {
                    "dimensions": (
                        "856x478"
                    ),
                    "frames": 453,
                    "rsz_observed_mib": 98,
                    "blr_observed_mib": 77,
                },
            },
            "important_note": (
                "FFV1 estimates are for "
                "disk-capacity planning only. "
                "Compression ratio is "
                "content-dependent and the pilot "
                "sizes were taken from rounded "
                "du -h observations."
            ),
        },
        "groups": {},
    }

    for key in sorted(
        totals
    ):
        summary[
            "groups"
        ][key] = (
            make_group_summary(
                totals[key]
            )
        )

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output_json.open(
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

    print(
        "STORAGE ESTIMATE COMPLETED"
    )
    print()

    print(
        "Manifest videos:       "
        "{}".format(
            summary[
                "manifest_videos"
            ]
        )
    )

    print(
        "Matched preflight:     "
        "{}".format(
            summary[
                "matched_preflight_videos"
            ]
        )
    )

    print()

    print(
        "Calibration ratios:"
    )

    print(
        "  FF++ RSZ:  "
        "{:.6f}".format(
            args.ffpp_rsz_ratio
        )
    )

    print(
        "  FF++ BLR:  "
        "{:.6f}".format(
            args.ffpp_blr_ratio
        )
    )

    print(
        "  Celeb RSZ: "
        "{:.6f}".format(
            args.celeb_rsz_ratio
        )
    )

    print(
        "  Celeb BLR: "
        "{:.6f}".format(
            args.celeb_blr_ratio
        )
    )

    print()

    display_groups = (
        "FaceForensics++|validation",
        "FaceForensics++|test",
        "Celeb-DF-v2|test",
        "FaceForensics++",
        "Celeb-DF-v2",
        "ALL",
    )

    for key in display_groups:
        group = (
            summary[
                "groups"
            ].get(key)
        )

        if group is None:
            continue

        print(
            "=== {} ===".format(
                key
            )
        )

        print(
            "Videos:               "
            "{}".format(
                group["videos"]
            )
        )

        print(
            "Decoded frames:       "
            "{}".format(
                group[
                    "decoded_frames"
                ]
            )
        )

        print(
            "Raw BGR:              "
            "{:.2f} GiB".format(
                group[
                    "raw_bgr_gib"
                ]
            )
        )

        print(
            "RSZ estimate:         "
            "{:.2f} GiB".format(
                group[
                    "rsz_estimated_gib"
                ]
            )
        )

        print(
            "BLR estimate:         "
            "{:.2f} GiB".format(
                group[
                    "blr_estimated_gib"
                ]
            )
        )

        print(
            "RSZ + BLR estimate:   "
            "{:.2f} GiB".format(
                group[
                    "rsz_plus_blr_estimated_gib"
                ]
            )
        )

        print(
            "With 25% headroom:    "
            "{:.2f} GiB".format(
                group[
                    "rsz_plus_blr_with_25pct_headroom_gib"
                ]
            )
        )

        print()

    print(
        "Output: {}".format(
            args.output_json
        )
    )


if __name__ == "__main__":
    main()