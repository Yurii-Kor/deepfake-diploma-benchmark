from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
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

DEFAULT_SNAPSHOT_PATH = (
    Path.home()
    / "deepfake_lab"
    / "study_data"
    / "provenance"
    / "environment_snapshot.json"
)

WRITER_PATH = (
    REPO_ROOT
    / "study"
    / "reproducibility"
    / "write_environment_snapshot.py"
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


def read_json(
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

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        value = json.load(
            file
        )

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            "Environment snapshot must contain "
            "a top-level JSON object."
        )

    return value


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
        stderr=subprocess.PIPE,
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
        raise ValueError(
            "Command produced no output: {}".format(
                " ".join(
                    arguments
                )
            )
        )

    return lines[0]


def strict_int(
    value: Any,
    *,
    field_name: str,
) -> int:
    if type(value) is not int:
        raise ValueError(
            "{} must be an exact integer; got {!r}."
            .format(
                field_name,
                value,
            )
        )

    return value


def strict_bool(
    value: Any,
    *,
    field_name: str,
) -> bool:
    if type(value) is not bool:
        raise ValueError(
            "{} must be an exact boolean; got {!r}."
            .format(
                field_name,
                value,
            )
        )

    return value


def require_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if (
        type(value) is not str
        or not value
    ):
        raise ValueError(
            "{} must be a non-empty string."
            .format(
                field_name
            )
        )

    return value


def current_repository() -> Dict[str, Any]:
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


def current_cuda() -> Dict[str, Any]:
    available = bool(
        torch.cuda.is_available()
    )

    result = {
        "available": available,
        "pytorch_cuda": (
            torch.version.cuda
        ),
        "cudnn_version": None,
        "device_count": 0,
        "devices": [],
    }

    if not available:
        return result

    cudnn_version = (
        torch.backends.cudnn.version()
    )

    result[
        "cudnn_version"
    ] = (
        int(
            cudnn_version
        )
        if cudnn_version is not None
        else None
    )

    device_count = int(
        torch.cuda.device_count()
    )

    result[
        "device_count"
    ] = device_count

    devices = []

    for index in range(
        device_count
    ):
        properties = (
            torch.cuda.get_device_properties(
                index
            )
        )

        devices.append(
            {
                "index": index,

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

    result[
        "devices"
    ] = devices

    return result


def validate_environment_snapshot(
    snapshot_path: Path,
) -> Dict[str, Any]:
    snapshot_path = (
        snapshot_path
        .expanduser()
        .resolve()
    )

    snapshot = read_json(
        snapshot_path
    )

    schema_version = strict_int(
        snapshot.get(
            "schema_version"
        ),
        field_name="schema_version",
    )

    if schema_version != 1:
        raise ValueError(
            "Unexpected environment snapshot "
            "schema version."
        )

    if (
        snapshot.get(
            "artifact_type"
        )
        != "environment_snapshot"
    ):
        raise ValueError(
            "Unexpected environment snapshot "
            "artifact type."
        )

    created_utc = require_string(
        snapshot.get(
            "created_utc"
        ),
        field_name="created_utc",
    )

    try:
        created = datetime.fromisoformat(
            created_utc
        )

    except ValueError as exc:
        raise ValueError(
            "created_utc is not a valid "
            "ISO-8601 timestamp."
        ) from exc

    if (
        created.tzinfo is None
        or created.utcoffset()
        is None
    ):
        raise ValueError(
            "created_utc must contain timezone "
            "information."
        )

    repository = snapshot.get(
        "repository"
    )

    if not isinstance(
        repository,
        dict,
    ):
        raise ValueError(
            "Environment snapshot has no "
            "repository metadata."
        )

    require_string(
        repository.get(
            "root"
        ),
        field_name="repository.root",
    )

    require_string(
        repository.get(
            "commit"
        ),
        field_name="repository.commit",
    )

    require_string(
        repository.get(
            "branch"
        ),
        field_name="repository.branch",
    )

    strict_bool(
        repository.get(
            "working_tree_dirty"
        ),
        field_name=(
            "repository.working_tree_dirty"
        ),
    )

    actual_repository = (
        current_repository()
    )

    if repository != actual_repository:
        raise ValueError(
            "Environment snapshot repository state "
            "differs from the current repository."
        )

    python_metadata = snapshot.get(
        "python"
    )

    if not isinstance(
        python_metadata,
        dict,
    ):
        raise ValueError(
            "Environment snapshot has no "
            "Python metadata."
        )

    expected_python = {
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
    }

    if (
        python_metadata
        != expected_python
    ):
        raise ValueError(
            "Environment snapshot Python metadata "
            "differs from the current runtime."
        )

    platform_metadata = snapshot.get(
        "platform"
    )

    if not isinstance(
        platform_metadata,
        dict,
    ):
        raise ValueError(
            "Environment snapshot has no "
            "platform metadata."
        )

    expected_platform = {
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
    }

    if (
        platform_metadata
        != expected_platform
    ):
        raise ValueError(
            "Environment snapshot platform metadata "
            "differs from the current runtime."
        )

    libraries = snapshot.get(
        "libraries"
    )

    if not isinstance(
        libraries,
        dict,
    ):
        raise ValueError(
            "Environment snapshot has no "
            "library metadata."
        )

    expected_libraries = {
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
    }

    if (
        libraries
        != expected_libraries
    ):
        raise ValueError(
            "Environment snapshot library versions "
            "differ from the current runtime."
        )

    cuda = snapshot.get(
        "cuda"
    )

    if not isinstance(
        cuda,
        dict,
    ):
        raise ValueError(
            "Environment snapshot has no "
            "CUDA metadata."
        )

    strict_bool(
        cuda.get(
            "available"
        ),
        field_name="cuda.available",
    )

    strict_int(
        cuda.get(
            "device_count"
        ),
        field_name="cuda.device_count",
    )

    if (
        cuda.get(
            "cudnn_version"
        )
        is not None
    ):
        strict_int(
            cuda.get(
                "cudnn_version"
            ),
            field_name="cuda.cudnn_version",
        )

    actual_cuda = (
        current_cuda()
    )

    if cuda != actual_cuda:
        raise ValueError(
            "Environment snapshot CUDA metadata "
            "differs from the current runtime."
        )

    external_tools = snapshot.get(
        "external_tools"
    )

    if not isinstance(
        external_tools,
        dict,
    ):
        raise ValueError(
            "Environment snapshot has no "
            "external-tool metadata."
        )

    expected_external_tools = {
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
    }

    if (
        external_tools
        != expected_external_tools
    ):
        raise ValueError(
            "Environment snapshot FFmpeg/FFprobe "
            "metadata differs from the current toolchain."
        )

    implementation = snapshot.get(
        "implementation"
    )

    if not isinstance(
        implementation,
        dict,
    ):
        raise ValueError(
            "Environment snapshot has no "
            "implementation fingerprint."
        )

    if (
        set(
            implementation
        )
        != {
            "write_environment_snapshot.py"
        }
    ):
        raise ValueError(
            "Environment snapshot implementation "
            "set mismatch."
        )

    writer_record = (
        implementation[
            "write_environment_snapshot.py"
        ]
    )

    if not isinstance(
        writer_record,
        dict,
    ):
        raise ValueError(
            "Environment writer fingerprint "
            "must be an object."
        )

    actual_writer = (
        file_record(
            WRITER_PATH
        )
    )

    if (
        writer_record
        != actual_writer
    ):
        raise ValueError(
            "Environment snapshot writer "
            "fingerprint mismatch."
        )

    return {
        "snapshot": (
            file_record(
                snapshot_path
            )
        ),

        "created_utc": (
            created_utc
        ),

        "repository": (
            repository
        ),

        "python": (
            python_metadata
        ),

        "platform": (
            platform_metadata
        ),

        "libraries": (
            libraries
        ),

        "cuda": (
            cuda
        ),

        "external_tools": (
            external_tools
        ),

        "writer_implementation": (
            actual_writer
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen environment "
            "snapshot."
        )
    )

    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT_PATH,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    result = (
        validate_environment_snapshot(
            args.snapshot
        )
    )

    print(
        "ENVIRONMENT SNAPSHOT VALIDATION"
    )

    print(
        "  Python:              {}".format(
            result[
                "python"
            ][
                "version_short"
            ]
        )
    )

    print(
        "  NumPy:               {}".format(
            result[
                "libraries"
            ][
                "numpy"
            ]
        )
    )

    print(
        "  PyTorch:             {}".format(
            result[
                "libraries"
            ][
                "pytorch"
            ]
        )
    )

    print(
        "  OpenCV:              {}".format(
            result[
                "libraries"
            ][
                "opencv"
            ]
        )
    )

    print(
        "  PyYAML:              {}".format(
            result[
                "libraries"
            ][
                "pyyaml"
            ]
        )
    )

    print(
        "  CUDA available:      {}".format(
            result[
                "cuda"
            ][
                "available"
            ]
        )
    )

    print(
        "  PyTorch CUDA:        {}".format(
            result[
                "cuda"
            ][
                "pytorch_cuda"
            ]
        )
    )

    print(
        "  cuDNN:               {}".format(
            result[
                "cuda"
            ][
                "cudnn_version"
            ]
        )
    )

    if (
        result[
            "cuda"
        ][
            "devices"
        ]
    ):
        print(
            "  primary GPU:         {}".format(
                result[
                    "cuda"
                ][
                    "devices"
                ][0][
                    "name"
                ]
            )
        )

    print(
        "  Git commit:          {}".format(
            result[
                "repository"
            ][
                "commit"
            ]
        )
    )

    print(
        "  Git branch:          {}".format(
            result[
                "repository"
            ][
                "branch"
            ]
        )
    )

    print(
        "  working tree dirty:  {}".format(
            result[
                "repository"
            ][
                "working_tree_dirty"
            ]
        )
    )

    print(
        "  snapshot SHA-256:    {}".format(
            result[
                "snapshot"
            ][
                "sha256"
            ]
        )
    )

    print()

    print(
        "ENVIRONMENT SNAPSHOT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()