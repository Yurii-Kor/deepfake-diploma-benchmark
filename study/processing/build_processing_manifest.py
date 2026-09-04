from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


FFPP_MANIPULATION_DIRS = (
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
)

CONDITIONS = (
    "RSZ",
    "BLR",
    "H40",
    "PLT",
)


def load_ffpp_pairs(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "FF++ split must contain a list: {}".format(path)
        )

    pairs = []

    for index, pair in enumerate(data, start=1):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
        ):
            raise ValueError(
                "Invalid FF++ pair at position {} in {}: {!r}".format(
                    index,
                    path,
                    pair,
                )
            )

        first = str(pair[0])
        second = str(pair[1])

        pairs.append((first, second))

    return pairs


def check_ffpp_split_overlap(split_pairs):
    split_ids = {}

    for split_name, pairs in split_pairs.items():
        flat = [
            video_id
            for pair in pairs
            for video_id in pair
        ]

        ids = set(flat)

        if len(flat) != len(ids):
            raise ValueError(
                "Duplicate FF++ source IDs within split '{}'."
                .format(split_name)
            )

        split_ids[split_name] = ids

    names = sorted(split_ids)

    for i, first in enumerate(names):
        for second in names[i + 1:]:
            overlap = (
                split_ids[first]
                & split_ids[second]
            )

            if overlap:
                raise ValueError(
                    "FF++ split overlap between {} and {}: {}"
                    .format(
                        first,
                        second,
                        sorted(overlap),
                    )
                )


def build_ffpp_records(
    ffpp_root,
    split_name,
    pairs,
    include_manipulations,
):
    records = []

    original_ids = sorted(
        {
            video_id
            for pair in pairs
            for video_id in pair
        }
    )

    for video_id in original_ids:
        relative_path = Path("original") / (
            "{}.mp4".format(video_id)
        )

        absolute_path = (
            ffpp_root
            / relative_path
        )

        records.append(
            {
                "dataset": "FaceForensics++",
                "role": split_name,
                "subgroup": "original",
                "study_label": 0,
                "source_label": "",
                "base_video_id": video_id,
                "relative_source_path": str(relative_path),
                "absolute_source_path": str(absolute_path),
                "conditions": ";".join(CONDITIONS),
            }
        )

    if include_manipulations:
        for subgroup in FFPP_MANIPULATION_DIRS:
            for first, second in pairs:
                for source_id, target_id in (
                    (first, second),
                    (second, first),
                ):
                    stem = "{}_{}".format(
                        source_id,
                        target_id,
                    )

                    relative_path = (
                        Path(subgroup)
                        / "{}.mp4".format(stem)
                    )

                    absolute_path = (
                        ffpp_root
                        / relative_path
                    )

                    records.append(
                        {
                            "dataset": "FaceForensics++",
                            "role": split_name,
                            "subgroup": subgroup,
                            "study_label": 1,
                            "source_label": "",
                            "base_video_id": stem,
                            "relative_source_path": str(relative_path),
                            "absolute_source_path": str(absolute_path),
                            "conditions": ";".join(CONDITIONS),
                        }
                    )

    return records


def load_celeb_test_records(
    celeb_root,
    list_path,
):
    records = []

    seen_paths = set()

    with list_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            parts = stripped.split(
                maxsplit=1
            )

            if len(parts) != 2:
                raise ValueError(
                    "Invalid Celeb-DF list line {}: {!r}"
                    .format(
                        line_number,
                        stripped,
                    )
                )

            source_label_text, relative_text = parts

            if source_label_text not in ("0", "1"):
                raise ValueError(
                    "Unexpected Celeb-DF source label "
                    "at line {}: {}"
                    .format(
                        line_number,
                        source_label_text,
                    )
                )

            if relative_text in seen_paths:
                raise ValueError(
                    "Duplicate Celeb-DF test path: {}"
                    .format(relative_text)
                )

            seen_paths.add(
                relative_text
            )

            relative_path = Path(
                relative_text
            )

            if len(relative_path.parts) < 2:
                raise ValueError(
                    "Invalid Celeb-DF relative path: {}"
                    .format(relative_text)
                )

            subgroup = (
                relative_path.parts[0]
            )

            source_label = int(
                source_label_text
            )

            # Official Celeb-DF convention:
            #   1 = real
            #   0 = fake
            #
            # Final study convention:
            #   0 = real
            #   1 = fake
            study_label = (
                0
                if source_label == 1
                else 1
            )

            absolute_path = (
                celeb_root
                / relative_path
            )

            records.append(
                {
                    "dataset": "Celeb-DF-v2",
                    "role": "test",
                    "subgroup": subgroup,
                    "study_label": study_label,
                    "source_label": source_label,
                    "base_video_id": relative_path.stem,
                    "relative_source_path": str(relative_path),
                    "absolute_source_path": str(absolute_path),
                    "conditions": ";".join(CONDITIONS),
                }
            )

    return records


def validate_files(records):
    missing = []

    for record in records:
        path = Path(
            record["absolute_source_path"]
        )

        if not path.is_file():
            missing.append(
                {
                    "dataset": record["dataset"],
                    "role": record["role"],
                    "relative_source_path": (
                        record[
                            "relative_source_path"
                        ]
                    ),
                }
            )

    return missing


def validate_unique_records(records):
    keys = Counter(
        (
            record["dataset"],
            record["role"],
            record["relative_source_path"],
        )
        for record in records
    )

    duplicates = [
        key
        for key, count in keys.items()
        if count != 1
    ]

    if duplicates:
        raise ValueError(
            "Duplicate processing-manifest records: {}"
            .format(duplicates[:20])
        )


def write_manifest(records, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "dataset",
        "role",
        "subgroup",
        "study_label",
        "source_label",
        "base_video_id",
        "relative_source_path",
        "absolute_source_path",
        "conditions",
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
        writer.writerows(records)


def make_summary(records):
    by_dataset = Counter(
        record["dataset"]
        for record in records
    )

    by_dataset_role = Counter(
        (
            record["dataset"],
            record["role"],
        )
        for record in records
    )

    by_dataset_role_subgroup = Counter(
        (
            record["dataset"],
            record["role"],
            record["subgroup"],
        )
        for record in records
    )

    label_counts = Counter(
        int(record["study_label"])
        for record in records
    )

    return {
        "total_base_videos": len(records),
        "conditions": list(CONDITIONS),
        "processed_outputs_per_base_video": len(CONDITIONS),
        "total_planned_processed_outputs": (
            len(records)
            * len(CONDITIONS)
        ),
        "study_label_counts": {
            str(key): value
            for key, value in sorted(
                label_counts.items()
            )
        },
        "dataset_counts": {
            key: value
            for key, value in sorted(
                by_dataset.items()
            )
        },
        "dataset_role_counts": {
            "{}|{}".format(
                dataset,
                role,
            ): value
            for (
                dataset,
                role,
            ), value in sorted(
                by_dataset_role.items()
            )
        },
        "dataset_role_subgroup_counts": {
            "{}|{}|{}".format(
                dataset,
                role,
                subgroup,
            ): value
            for (
                dataset,
                role,
                subgroup,
            ), value in sorted(
                by_dataset_role_subgroup.items()
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen processing manifest "
            "for FF++ validation/test and the "
            "official Celeb-DF-v2 test set."
        )
    )

    parser.add_argument(
        "--ffpp-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--ffpp-splits-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--celeb-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--celeb-test-list",
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

    args = parser.parse_args()

    if not args.ffpp_root.is_dir():
        raise FileNotFoundError(
            "FF++ root does not exist: {}"
            .format(args.ffpp_root)
        )

    if not args.ffpp_splits_dir.is_dir():
        raise FileNotFoundError(
            "FF++ splits directory does not exist: {}"
            .format(args.ffpp_splits_dir)
        )

    if not args.celeb_root.is_dir():
        raise FileNotFoundError(
            "Celeb-DF-v2 root does not exist: {}"
            .format(args.celeb_root)
        )

    if not args.celeb_test_list.is_file():
        raise FileNotFoundError(
            "Celeb-DF-v2 test list does not exist: {}"
            .format(args.celeb_test_list)
        )

    split_pairs = {
        split: load_ffpp_pairs(
            args.ffpp_splits_dir
            / "{}.json".format(split)
        )
        for split in (
            "train",
            "val",
            "test",
        )
    }

    check_ffpp_split_overlap(
        split_pairs
    )

    records = []

    records.extend(
        build_ffpp_records(
            ffpp_root=args.ffpp_root,
            split_name="validation",
            pairs=split_pairs["val"],
            include_manipulations=False,
        )
    )

    records.extend(
        build_ffpp_records(
            ffpp_root=args.ffpp_root,
            split_name="test",
            pairs=split_pairs["test"],
            include_manipulations=True,
        )
    )

    records.extend(
        load_celeb_test_records(
            celeb_root=args.celeb_root,
            list_path=args.celeb_test_list,
        )
    )

    validate_unique_records(
        records
    )

    missing = validate_files(
        records
    )

    if missing:
        print(
            "MISSING SOURCE VIDEOS: {}"
            .format(len(missing))
        )

        for item in missing[:50]:
            print(
                "{} | {} | {}"
                .format(
                    item["dataset"],
                    item["role"],
                    item[
                        "relative_source_path"
                    ],
                )
            )

        if len(missing) > 50:
            print(
                "... {} additional missing files"
                .format(
                    len(missing) - 50
                )
            )

        raise RuntimeError(
            "Processing manifest cannot be frozen "
            "because source videos are missing."
        )

    records.sort(
        key=lambda record: (
            record["dataset"],
            record["role"],
            record["subgroup"],
            record["relative_source_path"],
        )
    )

    write_manifest(
        records=records,
        output_path=args.output_csv,
    )

    summary = make_summary(
        records
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

    print("PROCESSING MANIFEST BUILT")
    print()
    print(
        "Base videos:              {}"
        .format(
            summary[
                "total_base_videos"
            ]
        )
    )
    print(
        "Conditions/video:         {}"
        .format(
            summary[
                "processed_outputs_per_base_video"
            ]
        )
    )
    print(
        "Planned processed files:  {}"
        .format(
            summary[
                "total_planned_processed_outputs"
            ]
        )
    )
    print()
    print(
        "FF++ validation:          {}"
        .format(
            summary[
                "dataset_role_counts"
            ].get(
                "FaceForensics++|validation",
                0,
            )
        )
    )
    print(
        "FF++ test:                {}"
        .format(
            summary[
                "dataset_role_counts"
            ].get(
                "FaceForensics++|test",
                0,
            )
        )
    )
    print(
        "Celeb-DF-v2 test:         {}"
        .format(
            summary[
                "dataset_role_counts"
            ].get(
                "Celeb-DF-v2|test",
                0,
            )
        )
    )
    print()
    print(
        "Manifest: {}"
        .format(args.output_csv)
    )
    print(
        "Summary:  {}"
        .format(args.output_summary)
    )


if __name__ == "__main__":
    main()