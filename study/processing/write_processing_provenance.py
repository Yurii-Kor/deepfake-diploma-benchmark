from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def command_output(command):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    output = (
        result.stdout.strip()
        or result.stderr.strip()
    )

    return {
        "returncode": result.returncode,
        "output": output,
    }


def first_line(command):
    result = command_output(command)

    if not result["output"]:
        return None

    return result["output"].splitlines()[0]


def git_information(repo_root):
    commit = command_output(
        [
            "git",
            "-C",
            str(repo_root),
            "rev-parse",
            "HEAD",
        ]
    )

    status = command_output(
        [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain",
        ]
    )

    branch = command_output(
        [
            "git",
            "-C",
            str(repo_root),
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ]
    )

    return {
        "commit": (
            commit["output"]
            if commit["returncode"] == 0
            else None
        ),
        "branch": (
            branch["output"]
            if branch["returncode"] == 0
            else None
        ),
        "working_tree_dirty": (
            bool(status["output"])
            if status["returncode"] == 0
            else None
        ),
    }


def manifest_summary(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    dataset_counts = Counter(
        row["dataset"]
        for row in rows
    )

    role_counts = Counter(
        (
            row["dataset"],
            row["role"],
        )
        for row in rows
    )

    subgroup_counts = Counter(
        (
            row["dataset"],
            row["role"],
            row["subgroup"],
        )
        for row in rows
    )

    label_counts = Counter(
        row["study_label"]
        for row in rows
    )

    return {
        "records": len(rows),
        "dataset_counts": {
            key: value
            for key, value in sorted(
                dataset_counts.items()
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
                role_counts.items()
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
                subgroup_counts.items()
            )
        },
        "study_label_counts": {
            key: value
            for key, value in sorted(
                label_counts.items()
            )
        },
    }


def file_record(path):
    if not path.is_file():
        raise FileNotFoundError(
            "Required provenance file does not exist: {}".format(
                path
            )
        )

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Write a reproducibility snapshot "
            "for the frozen study processing setup."
        )
    )

    parser.add_argument(
        "--repo-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--ffpp-splits-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--celeb-test-list",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    repo_root = args.repo_root.resolve()

    processing_root = (
        repo_root
        / "study"
        / "processing"
    )

    tracked_processing_files = [
        processing_root
        / "processing_config.yaml",

        processing_root
        / "processing_common.py",

        processing_root
        / "generate_processed_video.py",

        processing_root
        / "validate_processed_video.py",

        processing_root
        / "preflight_source_videos.py",

        processing_root
        / "preflight_listed_videos.py",

        processing_root
        / "build_processing_manifest.py",

        processing_root
        / "estimate_processing_storage.py",

        processing_root
        / "run_processing_corpus.py",
    ]

    split_files = [
        args.ffpp_splits_dir
        / "train.json",

        args.ffpp_splits_dir
        / "val.json",

        args.ffpp_splits_dir
        / "test.json",
    ]

    snapshot = {
        "schema_version": 1,

        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "repository": {
            "root": str(repo_root),
            **git_information(
                repo_root
            ),
        },

        "runtime": {
            "python": (
                sys.version
            ),
            "python_executable": (
                sys.executable
            ),
            "platform": (
                platform.platform()
            ),
            "opencv": (
                cv2.__version__
            ),
            "numpy": (
                np.__version__
            ),
            "pyyaml": (
                yaml.__version__
            ),
            "ffmpeg": first_line(
                [
                    "ffmpeg",
                    "-version",
                ]
            ),
            "ffprobe": first_line(
                [
                    "ffprobe",
                    "-version",
                ]
            ),
        },

        "frozen_inputs": {
            "processing_manifest": (
                file_record(
                    args.manifest
                )
            ),

            "ffpp_official_splits": {
                path.name: (
                    file_record(path)
                )
                for path in split_files
            },

            "celeb_df_v2_official_test_list": (
                file_record(
                    args.celeb_test_list
                )
            ),
        },

        "processing_implementation": {
            path.name: (
                file_record(path)
            )
            for path in (
                tracked_processing_files
            )
        },

        "manifest_summary": (
            manifest_summary(
                args.manifest
            )
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        args.output.with_name(
            args.output.name
            + ".tmp"
        )
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            snapshot,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    temp_path.replace(
        args.output
    )

    print(
        "PROCESSING PROVENANCE WRITTEN"
    )

    print(
        "Output: {}".format(
            args.output
        )
    )

    print(
        "Git commit: {}".format(
            snapshot[
                "repository"
            ][
                "commit"
            ]
        )
    )

    print(
        "Working tree dirty: {}".format(
            snapshot[
                "repository"
            ][
                "working_tree_dirty"
            ]
        )
    )

    print(
        "Manifest records: {}".format(
            snapshot[
                "manifest_summary"
            ][
                "records"
            ]
        )
    )

    print(
        "Manifest SHA-256: {}".format(
            snapshot[
                "frozen_inputs"
            ][
                "processing_manifest"
            ][
                "sha256"
            ]
        )
    )


if __name__ == "__main__":
    main()