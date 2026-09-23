from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import yaml


REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


DEFAULT_STUDY_ROOT = (
    Path.home()
    / "deepfake_lab"
    / "study_data"
)


def sha256_file(
    path: Path,
) -> str:
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


def run_command(
    arguments: List[str],
    *,
    cwd: Path = None,
) -> str:
    completed = subprocess.run(
        arguments,
        cwd=(
            str(cwd)
            if cwd is not None
            else None
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        check=True,
    )

    return completed.stdout.strip()


def first_line(
    arguments: List[str],
) -> str:
    output = run_command(
        arguments
    )

    lines = output.splitlines()

    if not lines:
        raise RuntimeError(
            "Command produced no output: {}".format(
                " ".join(
                    arguments
                )
            )
        )

    return lines[0]


def git_information() -> Dict[str, Any]:
    commit = run_command(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=REPO_ROOT,
    )

    branch = run_command(
        [
            "git",
            "branch",
            "--show-current",
        ],
        cwd=REPO_ROOT,
    )

    status = run_command(
        [
            "git",
            "status",
            "--porcelain",
        ],
        cwd=REPO_ROOT,
    )

    return {
        "root": str(
            REPO_ROOT
        ),
        "commit": commit,
        "branch": branch,
        "working_tree_dirty": bool(
            status
        ),
    }


def cuda_information() -> Dict[str, Any]:
    available = bool(
        torch.cuda.is_available()
    )

    information = {
        "available": available,
        "pytorch_cuda": (
            torch.version.cuda
        ),
        "cudnn_version": None,
        "device_count": 0,
        "devices": [],
    }

    if not available:
        return information

    cudnn_version = (
        torch.backends.cudnn.version()
    )

    information[
        "cudnn_version"
    ] = (
        int(
            cudnn_version
        )
        if cudnn_version is not None
        else None
    )

    device_count = (
        torch.cuda.device_count()
    )

    information[
        "device_count"
    ] = int(
        device_count
    )

    devices = []

    for device_index in range(
        device_count
    ):
        properties = (
            torch.cuda.get_device_properties(
                device_index
            )
        )

        devices.append(
            {
                "index": (
                    device_index
                ),
                "name": (
                    properties.name
                ),
                "total_memory_bytes": int(
                    properties.total_memory
                ),
                "compute_capability": (
                    "{}.{}".format(
                        properties.major,
                        properties.minor,
                    )
                ),
            }
        )

    information[
        "devices"
    ] = devices

    return information


def file_record(
    path: Path,
) -> Dict[str, Any]:
    path = (
        path
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    return {
        "path": str(
            path
        ),
        "sha256": (
            sha256_file(
                path
            )
        ),
        "size_bytes": (
            path.stat().st_size
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Write the frozen runtime/environment "
            "snapshot for the deepfake robustness study."
        )
    )

    parser.add_argument(
        "--study-root",
        type=Path,
        default=DEFAULT_STUDY_ROOT,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    study_root = (
        args.study_root
        .expanduser()
        .resolve()
    )

    output_path = (
        study_root
        / "provenance"
        / "environment_snapshot.json"
    )

    implementation_path = (
        Path(__file__)
        .resolve()
    )

    snapshot = {
        "schema_version": 1,

        "artifact_type": (
            "environment_snapshot"
        ),

        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "repository": (
            git_information()
        ),

        "python": {
            "version": (
                sys.version
            ),
            "version_short": (
                platform.python_version()
            ),
            "implementation": (
                platform.python_implementation()
            ),
            "executable": (
                sys.executable
            ),
        },

        "platform": {
            "platform": (
                platform.platform()
            ),
            "system": (
                platform.system()
            ),
            "release": (
                platform.release()
            ),
            "machine": (
                platform.machine()
            ),
        },

        "libraries": {
            "numpy": (
                np.__version__
            ),
            "pytorch": (
                torch.__version__
            ),
            "opencv": (
                cv2.__version__
            ),
            "pyyaml": (
                yaml.__version__
            ),
        },

        "cuda": (
            cuda_information()
        ),

        "external_tools": {
            "ffmpeg": (
                first_line(
                    [
                        "ffmpeg",
                        "-version",
                    ]
                )
            ),
            "ffprobe": (
                first_line(
                    [
                        "ffprobe",
                        "-version",
                    ]
                )
            ),
        },

        "implementation": {
            "write_environment_snapshot.py": (
                file_record(
                    implementation_path
                )
            ),
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        output_path.with_name(
            output_path.name
            + ".tmp"
        )
    )

    with temporary_path.open(
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

        file.write(
            "\n"
        )

    temporary_path.replace(
        output_path
    )

    print(
        "ENVIRONMENT SNAPSHOT WRITTEN"
    )

    print(
        "  output:              {}".format(
            output_path
        )
    )

    print(
        "  Python:              {}".format(
            snapshot[
                "python"
            ][
                "version_short"
            ]
        )
    )

    print(
        "  NumPy:               {}".format(
            snapshot[
                "libraries"
            ][
                "numpy"
            ]
        )
    )

    print(
        "  PyTorch:             {}".format(
            snapshot[
                "libraries"
            ][
                "pytorch"
            ]
        )
    )

    print(
        "  OpenCV:              {}".format(
            snapshot[
                "libraries"
            ][
                "opencv"
            ]
        )
    )

    print(
        "  PyYAML:              {}".format(
            snapshot[
                "libraries"
            ][
                "pyyaml"
            ]
        )
    )

    print(
        "  CUDA available:      {}".format(
            snapshot[
                "cuda"
            ][
                "available"
            ]
        )
    )

    print(
        "  PyTorch CUDA:        {}".format(
            snapshot[
                "cuda"
            ][
                "pytorch_cuda"
            ]
        )
    )

    print(
        "  cuDNN:               {}".format(
            snapshot[
                "cuda"
            ][
                "cudnn_version"
            ]
        )
    )

    if (
        snapshot[
            "cuda"
        ][
            "devices"
        ]
    ):
        print(
            "  primary GPU:         {}".format(
                snapshot[
                    "cuda"
                ][
                    "devices"
                ][0][
                    "name"
                ]
            )
        )

    print(
        "  FFmpeg:              {}".format(
            snapshot[
                "external_tools"
            ][
                "ffmpeg"
            ]
        )
    )

    print(
        "  FFprobe:             {}".format(
            snapshot[
                "external_tools"
            ][
                "ffprobe"
            ]
        )
    )

    print(
        "  Git commit:          {}".format(
            snapshot[
                "repository"
            ][
                "commit"
            ]
        )
    )

    print(
        "  Git branch:          {}".format(
            snapshot[
                "repository"
            ][
                "branch"
            ]
        )
    )

    print(
        "  working tree dirty:  {}".format(
            snapshot[
                "repository"
            ][
                "working_tree_dirty"
            ]
        )
    )

    print()
    print(
        "ENVIRONMENT SNAPSHOT PASSED"
    )


if __name__ == "__main__":
    main()