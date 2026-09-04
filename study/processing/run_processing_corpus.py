from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from processing_common import load_config


GIB = 1024 ** 3


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def read_manifest(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def record_matches_filters(
    record,
    dataset,
    role,
    subgroup,
):
    if (
        dataset is not None
        and record["dataset"] != dataset
    ):
        return False

    if (
        role is not None
        and record["role"] != role
    ):
        return False

    if (
        subgroup is not None
        and record["subgroup"] != subgroup
    ):
        return False

    return True


def output_path_for_record(
    output_root,
    record,
    condition,
    extension,
):
    relative_source = Path(
        record["relative_source_path"]
    )

    relative_output = (
        relative_source.with_suffix(
            extension
        )
    )

    return (
        output_root
        / record["dataset"]
        / condition
        / relative_output
    )


def run_command(command):
    started = time.monotonic()

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    elapsed = (
        time.monotonic()
        - started
    )

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed_seconds": elapsed,
    }


def append_jsonl(
    path,
    record,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        json.dump(
            record,
            file,
            ensure_ascii=False,
            sort_keys=True,
        )

        file.write("\n")
        file.flush()

        os.fsync(
            file.fileno()
        )


def tail_text(
    text,
    limit=6000,
):
    if text is None:
        return ""

    if len(text) <= limit:
        return text

    return text[-limit:]


def free_disk_gib(
    path,
):
    usage = shutil.disk_usage(
        str(path)
    )

    return (
        usage.free
        / GIB
    )


def make_base_qc_record(
    manifest_record,
    condition,
    source_path,
    output_path,
):
    return {
        "timestamp_utc": utc_now(),
        "dataset": (
            manifest_record["dataset"]
        ),
        "role": (
            manifest_record["role"]
        ),
        "subgroup": (
            manifest_record["subgroup"]
        ),
        "study_label": int(
            manifest_record["study_label"]
        ),
        "source_label": (
            manifest_record["source_label"]
        ),
        "base_video_id": (
            manifest_record["base_video_id"]
        ),
        "relative_source_path": (
            manifest_record[
                "relative_source_path"
            ]
        ),
        "source_path": str(
            source_path
        ),
        "condition": condition,
        "output_path": str(
            output_path
        ),
    }


def write_summary(
    path,
    summary,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        path.name + ".tmp"
    )

    with temp_path.open(
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

    os.replace(
        temp_path,
        path,
    )


def print_dry_run_summary(
    selected,
    condition,
    output_root,
    extension,
):
    print("PROCESSING DRY RUN")
    print()
    print(
        "Condition:       {}".format(
            condition
        )
    )
    print(
        "Selected videos: {}".format(
            len(selected)
        )
    )
    print(
        "Output root:     {}".format(
            output_root
        )
    )
    print()

    groups = Counter(
        (
            row["dataset"],
            row["role"],
            row["subgroup"],
        )
        for row in selected
    )

    print("Selected groups:")

    for key, count in sorted(
        groups.items()
    ):
        print(
            "  {:<20} {:<12} {:<18} {}".format(
                key[0],
                key[1],
                key[2],
                count,
            )
        )

    print()

    existing = 0
    missing = 0

    for row in selected:
        output_path = (
            output_path_for_record(
                output_root=output_root,
                record=row,
                condition=condition,
                extension=extension,
            )
        )

        if output_path.is_file():
            existing += 1
        else:
            missing += 1

    print(
        "Existing outputs: {}".format(
            existing
        )
    )
    print(
        "Missing outputs:  {}".format(
            missing
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate and validate one processing "
            "condition over the frozen study manifest."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--condition",
        required=True,
    )

    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--qc-jsonl",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--summary-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--dataset",
        default=None,
    )

    parser.add_argument(
        "--role",
        default=None,
    )

    parser.add_argument(
        "--subgroup",
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=0.0,
        help=(
            "Stop before generating a new video "
            "if free space falls below this value."
        ),
    )

    parser.add_argument(
        "--repair-invalid",
        action="store_true",
        help=(
            "Delete an existing output only after "
            "it fails validation, then regenerate it."
        ),
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError(
            "--limit must be greater than zero."
        )

    if args.min_free_gib < 0:
        raise ValueError(
            "--min-free-gib cannot be negative."
        )

    if not args.manifest.is_file():
        raise FileNotFoundError(
            "Manifest does not exist: {}".format(
                args.manifest
            )
        )

    config = load_config(
        args.config
    )

    if (
        args.condition
        not in config["conditions"]
    ):
        raise ValueError(
            "Unknown condition '{}'. "
            "Available conditions: {}".format(
                args.condition,
                sorted(
                    config["conditions"]
                ),
            )
        )

    condition_config = (
        config["conditions"][
            args.condition
        ]
    )

    extension = condition_config[
        "output_extension"
    ]

    manifest_rows = read_manifest(
        args.manifest
    )

    selected = [
        row
        for row in manifest_rows
        if record_matches_filters(
            record=row,
            dataset=args.dataset,
            role=args.role,
            subgroup=args.subgroup,
        )
    ]

    if args.limit is not None:
        selected = selected[
            :args.limit
        ]

    if not selected:
        raise RuntimeError(
            "No manifest records matched "
            "the requested filters."
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.dry_run:
        print_dry_run_summary(
            selected=selected,
            condition=args.condition,
            output_root=args.output_root,
            extension=extension,
        )

        return

    script_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    generator_script = (
        script_dir
        / "generate_processed_video.py"
    )

    validator_script = (
        script_dir
        / "validate_processed_video.py"
    )

    if not generator_script.is_file():
        raise FileNotFoundError(
            generator_script
        )

    if not validator_script.is_file():
        raise FileNotFoundError(
            validator_script
        )

    counters = Counter()

    run_started = utc_now()

    total_selected = len(
        selected
    )

    print(
        "PROCESSING RUN STARTED"
    )
    print()
    print(
        "Condition:       {}".format(
            args.condition
        )
    )
    print(
        "Selected videos: {}".format(
            total_selected
        )
    )
    print(
        "Output root:     {}".format(
            args.output_root
        )
    )
    print(
        "QC JSONL:        {}".format(
            args.qc_jsonl
        )
    )
    print(
        "Minimum free:    {:.2f} GiB".format(
            args.min_free_gib
        )
    )
    print()

    for index, manifest_record in enumerate(
        selected,
        start=1,
    ):
        source_path = Path(
            manifest_record[
                "absolute_source_path"
            ]
        )

        output_path = (
            output_path_for_record(
                output_root=args.output_root,
                record=manifest_record,
                condition=args.condition,
                extension=extension,
            )
        )

        base_qc = make_base_qc_record(
            manifest_record=manifest_record,
            condition=args.condition,
            source_path=source_path,
            output_path=output_path,
        )

        print(
            "[{}/{}] {} | {} | {}".format(
                index,
                total_selected,
                manifest_record[
                    "dataset"
                ],
                manifest_record[
                    "role"
                ],
                manifest_record[
                    "relative_source_path"
                ],
            )
        )

        if not source_path.is_file():
            record = dict(
                base_qc
            )

            record.update(
                {
                    "status": (
                        "source_missing"
                    ),
                    "error": (
                        "Source file does not exist."
                    ),
                }
            )

            append_jsonl(
                args.qc_jsonl,
                record,
            )

            counters[
                "source_missing"
            ] += 1

            print(
                "  FAILED: source missing"
            )

            if args.stop_on_error:
                break

            continue

        was_repair = False
        previous_validation_error = ""

        if output_path.is_file():
            validate_command = [
                sys.executable,
                str(
                    validator_script
                ),
                "--config",
                str(
                    args.config
                ),
                "--condition",
                args.condition,
                "--source",
                str(
                    source_path
                ),
                "--processed",
                str(
                    output_path
                ),
            ]

            validation = run_command(
                validate_command
            )

            if (
                validation[
                    "returncode"
                ] == 0
            ):
                record = dict(
                    base_qc
                )

                record.update(
                    {
                        "status": (
                            "existing_valid"
                        ),
                        "validation_seconds": (
                            round(
                                validation[
                                    "elapsed_seconds"
                                ],
                                3,
                            )
                        ),
                        "output_size_bytes": (
                            output_path.stat().st_size
                        ),
                        "free_disk_gib": round(
                            free_disk_gib(
                                args.output_root
                            ),
                            3,
                        ),
                    }
                )

                append_jsonl(
                    args.qc_jsonl,
                    record,
                )

                counters[
                    "existing_valid"
                ] += 1

                print(
                    "  SKIP: existing output valid"
                )

                continue

            previous_validation_error = (
                tail_text(
                    validation[
                        "stderr"
                    ]
                    or validation[
                        "stdout"
                    ]
                )
            )

            if not args.repair_invalid:
                record = dict(
                    base_qc
                )

                record.update(
                    {
                        "status": (
                            "existing_invalid"
                        ),
                        "error": (
                            previous_validation_error
                        ),
                        "validation_seconds": (
                            round(
                                validation[
                                    "elapsed_seconds"
                                ],
                                3,
                            )
                        ),
                        "output_size_bytes": (
                            output_path.stat().st_size
                        ),
                    }
                )

                append_jsonl(
                    args.qc_jsonl,
                    record,
                )

                counters[
                    "existing_invalid"
                ] += 1

                print(
                    "  FAILED: existing output invalid"
                )

                if previous_validation_error:
                    print(
                        "  {}".format(
                            previous_validation_error
                            .splitlines()[-1]
                        )
                    )

                if args.stop_on_error:
                    break

                continue

            print(
                "  Existing output invalid; "
                "repair requested."
            )

            output_path.unlink()

            was_repair = True

        current_free = free_disk_gib(
            args.output_root
        )

        if (
            args.min_free_gib > 0
            and current_free
            < args.min_free_gib
        ):
            record = dict(
                base_qc
            )

            record.update(
                {
                    "status": (
                        "low_disk_stop"
                    ),
                    "free_disk_gib": round(
                        current_free,
                        3,
                    ),
                    "minimum_free_gib": (
                        args.min_free_gib
                    ),
                }
            )

            append_jsonl(
                args.qc_jsonl,
                record,
            )

            counters[
                "low_disk_stop"
            ] += 1

            print(
                "  STOP: free disk {:.2f} GiB "
                "< minimum {:.2f} GiB".format(
                    current_free,
                    args.min_free_gib,
                )
            )

            break

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        generate_command = [
            sys.executable,
            str(
                generator_script
            ),
            "--config",
            str(
                args.config
            ),
            "--condition",
            args.condition,
            "--input",
            str(
                source_path
            ),
            "--output",
            str(
                output_path
            ),
        ]

        generation = run_command(
            generate_command
        )

        if (
            generation[
                "returncode"
            ] != 0
        ):
            record = dict(
                base_qc
            )

            record.update(
                {
                    "status": (
                        "generation_failed_after_repair"
                        if was_repair
                        else "generation_failed"
                    ),
                    "generation_seconds": (
                        round(
                            generation[
                                "elapsed_seconds"
                            ],
                            3,
                        )
                    ),
                    "error": tail_text(
                        generation[
                            "stderr"
                        ]
                        or generation[
                            "stdout"
                        ]
                    ),
                    "previous_validation_error": (
                        previous_validation_error
                    ),
                }
            )

            append_jsonl(
                args.qc_jsonl,
                record,
            )

            counters[
                record["status"]
            ] += 1

            print(
                "  FAILED: generation"
            )

            if record["error"]:
                print(
                    "  {}".format(
                        record[
                            "error"
                        ].splitlines()[-1]
                    )
                )

            if args.stop_on_error:
                break

            continue

        validate_command = [
            sys.executable,
            str(
                validator_script
            ),
            "--config",
            str(
                args.config
            ),
            "--condition",
            args.condition,
            "--source",
            str(
                source_path
            ),
            "--processed",
            str(
                output_path
            ),
        ]

        validation = run_command(
            validate_command
        )

        if (
            validation[
                "returncode"
            ] != 0
        ):
            record = dict(
                base_qc
            )

            record.update(
                {
                    "status": (
                        "generated_invalid"
                    ),
                    "generation_seconds": (
                        round(
                            generation[
                                "elapsed_seconds"
                            ],
                            3,
                        )
                    ),
                    "validation_seconds": (
                        round(
                            validation[
                                "elapsed_seconds"
                            ],
                            3,
                        )
                    ),
                    "error": tail_text(
                        validation[
                            "stderr"
                        ]
                        or validation[
                            "stdout"
                        ]
                    ),
                    "output_size_bytes": (
                        output_path.stat().st_size
                        if output_path.is_file()
                        else None
                    ),
                    "previous_validation_error": (
                        previous_validation_error
                    ),
                }
            )

            append_jsonl(
                args.qc_jsonl,
                record,
            )

            counters[
                "generated_invalid"
            ] += 1

            print(
                "  FAILED: generated output "
                "did not validate"
            )

            if record["error"]:
                print(
                    "  {}".format(
                        record[
                            "error"
                        ].splitlines()[-1]
                    )
                )

            if args.stop_on_error:
                break

            continue

        final_status = (
            "repaired_valid"
            if was_repair
            else "generated_valid"
        )

        record = dict(
            base_qc
        )

        record.update(
            {
                "status": final_status,
                "generation_seconds": round(
                    generation[
                        "elapsed_seconds"
                    ],
                    3,
                ),
                "validation_seconds": round(
                    validation[
                        "elapsed_seconds"
                    ],
                    3,
                ),
                "output_size_bytes": (
                    output_path.stat().st_size
                ),
                "free_disk_gib": round(
                    free_disk_gib(
                        args.output_root
                    ),
                    3,
                ),
                "previous_validation_error": (
                    previous_validation_error
                ),
            }
        )

        append_jsonl(
            args.qc_jsonl,
            record,
        )

        counters[
            final_status
        ] += 1

        print(
            "  OK: {} ({:.1f} MiB)".format(
                final_status,
                (
                    output_path.stat().st_size
                    / (1024 ** 2)
                ),
            )
        )

    run_finished = utc_now()

    summary = {
        "run_started_utc": (
            run_started
        ),
        "run_finished_utc": (
            run_finished
        ),
        "condition": (
            args.condition
        ),
        "dataset_filter": (
            args.dataset
        ),
        "role_filter": (
            args.role
        ),
        "subgroup_filter": (
            args.subgroup
        ),
        "limit": (
            args.limit
        ),
        "selected_videos": (
            total_selected
        ),
        "status_counts": {
            key: value
            for key, value in sorted(
                counters.items()
            )
        },
        "output_root": str(
            args.output_root
        ),
        "qc_jsonl": str(
            args.qc_jsonl
        ),
        "free_disk_gib_at_finish": round(
            free_disk_gib(
                args.output_root
            ),
            3,
        ),
    }

    write_summary(
        args.summary_json,
        summary,
    )

    print()
    print(
        "RUN FINISHED"
    )

    for status, count in sorted(
        counters.items()
    ):
        print(
            "  {:<32} {}".format(
                status,
                count,
            )
        )

    print()
    print(
        "Free disk: {:.2f} GiB".format(
            summary[
                "free_disk_gib_at_finish"
            ]
        )
    )

    print(
        "Summary:   {}".format(
            args.summary_json
        )
    )


if __name__ == "__main__":
    main()