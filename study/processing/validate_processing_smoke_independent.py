from __future__ import annotations

import csv
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np


STUDY_ROOT = (
    Path.home()
    / "deepfake_lab"
    / "study_data"
)

MANIFEST_PATH = (
    STUDY_ROOT
    / "manifests"
    / "processing_smoke_manifest.csv"
)

OUTPUT_ROOT = (
    STUDY_ROOT
    / "videos"
)

AUDIT_OUTPUT_PATH = (
    STUDY_ROOT
    / "qc"
    / "processing_smoke_independent_audit.json"
)

RATE_TOLERANCE_FPS = 0.001

EXPECTED_CONDITIONS = {
    "RSZ": {
        "extension": ".mkv",
        "codec": "ffv1",
        "pixel_format": "bgr0",
        "pixel_exact": True,
    },
    "BLR": {
        "extension": ".mkv",
        "codec": "ffv1",
        "pixel_format": "bgr0",
        "pixel_exact": True,
    },
    "H40": {
        "extension": ".mp4",
        "codec": "h264",
        "pixel_format": "yuv420p",
        "pixel_exact": False,
    },
    "PLT": {
        "extension": ".mp4",
        "codec": "h264",
        "pixel_format": "yuv420p",
        "pixel_exact": False,
    },
}


def run_json_command(command):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n{}\n\n{}".format(
                " ".join(
                    str(item)
                    for item in command
                ),
                result.stderr.strip(),
            )
        )

    try:
        return json.loads(
            result.stdout
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Command did not return valid JSON:\n{}".format(
                " ".join(
                    str(item)
                    for item in command
                )
            )
        ) from exc


def probe_video(path):
    data = run_json_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            (
                "stream="
                "codec_name,"
                "pix_fmt,"
                "width,"
                "height,"
                "r_frame_rate,"
                "avg_frame_rate,"
                "nb_read_frames"
            ),
            "-of",
            "json",
            str(path),
        ]
    )

    streams = data.get(
        "streams",
        [],
    )

    if len(streams) != 1:
        raise RuntimeError(
            "Expected exactly one selected video "
            "stream in {}.".format(
                path
            )
        )

    stream = streams[0]

    raw_count = stream.get(
        "nb_read_frames"
    )

    if raw_count in (
        None,
        "",
        "N/A",
    ):
        raise RuntimeError(
            "ffprobe did not report decoded "
            "frame count for {}.".format(
                path
            )
        )

    stream[
        "decoded_frame_count"
    ] = int(
        raw_count
    )

    return stream


def count_audio_streams(path):
    data = run_json_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            str(path),
        ]
    )

    return len(
        data.get(
            "streams",
            [],
        )
    )


def parse_rate(value):
    try:
        rate = Fraction(
            str(value)
        )
    except (
        ValueError,
        ZeroDivisionError,
    ):
        return None

    if rate <= 0:
        return None

    return rate


def rate_difference(
    first,
    second,
):
    first_rate = parse_rate(
        first
    )

    second_rate = parse_rate(
        second
    )

    if (
        first_rate is None
        or second_rate is None
    ):
        raise RuntimeError(
            "Unable to parse frame rates: "
            "{} / {}".format(
                first,
                second,
            )
        )

    return abs(
        float(
            first_rate
        )
        - float(
            second_rate
        )
    )


class RawBgrReader:
    def __init__(
        self,
        path,
        width,
        height,
    ):
        self.path = Path(
            path
        )

        self.width = int(
            width
        )

        self.height = int(
            height
        )

        self.frame_bytes = (
            self.width
            * self.height
            * 3
        )

        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(
                    self.path
                ),
                "-map",
                "0:v:0",
                "-an",
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if (
            self.process.stdout is None
            or self.process.stderr is None
        ):
            raise RuntimeError(
                "Unable to open FFmpeg pipes."
            )

        self.finished = False

    def _read_exact(
        self,
        size,
    ):
        chunks = []
        total = 0

        while total < size:
            chunk = (
                self.process.stdout.read(
                    size - total
                )
            )

            if not chunk:
                break

            chunks.append(
                chunk
            )

            total += len(
                chunk
            )

        return b"".join(
            chunks
        )

    def read_frame(self):
        if self.finished:
            return None

        raw = self._read_exact(
            self.frame_bytes
        )

        if len(raw) == 0:
            self._finish()
            return None

        if (
            len(raw)
            != self.frame_bytes
        ):
            self._finish(
                force=True
            )

            raise RuntimeError(
                "Partial raw frame from {}: "
                "{} / {} bytes.".format(
                    self.path,
                    len(
                        raw
                    ),
                    self.frame_bytes,
                )
            )

        frame = np.frombuffer(
            raw,
            dtype=np.uint8,
        )

        return frame.reshape(
            (
                self.height,
                self.width,
                3,
            )
        )

    def _finish(
        self,
        force=False,
    ):
        if self.finished:
            return

        self.finished = True

        if (
            force
            and self.process.poll()
            is None
        ):
            self.process.kill()

        return_code = (
            self.process.wait()
        )

        stderr = (
            self.process.stderr
            .read()
            .decode(
                errors="replace"
            )
            .strip()
        )

        self.process.stdout.close()
        self.process.stderr.close()

        if (
            not force
            and return_code != 0
        ):
            raise RuntimeError(
                "FFmpeg decode failed for {}:\n{}".format(
                    self.path,
                    stderr,
                )
            )

    def close(self):
        if not self.finished:
            if self.process.poll() is None:
                self.process.kill()

            self.process.wait()

            self.process.stdout.close()
            self.process.stderr.close()

            self.finished = True


def scaled_dimension(
    size,
    scale,
):
    return max(
        1,
        int(
            math.floor(
                size
                * scale
                + 0.5
            )
        ),
    )


def independent_rsz(
    frame,
):
    height, width = (
        frame.shape[
            :2
        ]
    )

    scaled_width = (
        scaled_dimension(
            width,
            0.5,
        )
    )

    scaled_height = (
        scaled_dimension(
            height,
            0.5,
        )
    )

    reduced = cv2.resize(
        frame,
        (
            scaled_width,
            scaled_height,
        ),
        interpolation=(
            cv2.INTER_CUBIC
        ),
    )

    return cv2.resize(
        reduced,
        (
            width,
            height,
        ),
        interpolation=(
            cv2.INTER_CUBIC
        ),
    )


def independent_blr(
    frame,
):
    return cv2.GaussianBlur(
        frame,
        (
            7,
            7,
        ),
        sigmaX=1.0,
        sigmaY=1.0,
        borderType=(
            cv2.BORDER_REFLECT_101
        ),
    )


def pixel_exact_audit(
    source_path,
    processed_path,
    condition,
    width,
    height,
):
    if condition == "RSZ":
        transform = (
            independent_rsz
        )

    elif condition == "BLR":
        transform = (
            independent_blr
        )

    else:
        raise ValueError(
            "Pixel-exact audit is defined "
            "only for RSZ and BLR."
        )

    source_reader = (
        RawBgrReader(
            source_path,
            width,
            height,
        )
    )

    processed_reader = (
        RawBgrReader(
            processed_path,
            width,
            height,
        )
    )

    frame_count = 0
    max_absolute_difference = 0

    try:
        while True:
            source_frame = (
                source_reader
                .read_frame()
            )

            processed_frame = (
                processed_reader
                .read_frame()
            )

            if (
                source_frame is None
                and processed_frame is None
            ):
                break

            if (
                source_frame is None
                or processed_frame is None
            ):
                raise RuntimeError(
                    "Independent raw decode found "
                    "a frame-count mismatch."
                )

            expected = transform(
                source_frame
            )

            difference = np.abs(
                expected.astype(
                    np.int16
                )
                - processed_frame.astype(
                    np.int16
                )
            )

            current_max = int(
                difference.max()
            )

            max_absolute_difference = max(
                max_absolute_difference,
                current_max,
            )

            if current_max != 0:
                raise RuntimeError(
                    "{} pixel mismatch at frame {}: "
                    "max absolute difference {}."
                    .format(
                        condition,
                        frame_count,
                        current_max,
                    )
                )

            frame_count += 1

    finally:
        source_reader.close()
        processed_reader.close()

    if frame_count == 0:
        raise RuntimeError(
            "Independent pixel audit "
            "decoded zero frames."
        )

    return {
        "decoded_frames": (
            frame_count
        ),
        "max_absolute_difference": (
            max_absolute_difference
        ),
    }


def read_manifest():
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(
            csv.DictReader(
                file
            )
        )


def processed_path(
    record,
    condition,
):
    spec = (
        EXPECTED_CONDITIONS[
            condition
        ]
    )

    relative = Path(
        record[
            "relative_source_path"
        ]
    ).with_suffix(
        spec[
            "extension"
        ]
    )

    return (
        OUTPUT_ROOT
        / record[
            "dataset"
        ]
        / condition
        / relative
    )


def main():
    manifest = (
        read_manifest()
    )

    if len(manifest) != 4:
        raise RuntimeError(
            "Expected exactly 4 smoke "
            "manifest records."
        )

    results = []

    pixel_exact_checks = 0

    print(
        "INDEPENDENT PROCESSING SMOKE AUDIT"
    )
    print()

    for record in manifest:
        source_path = Path(
            record[
                "absolute_source_path"
            ]
        )

        if not source_path.is_file():
            raise FileNotFoundError(
                source_path
            )

        source_probe = (
            probe_video(
                source_path
            )
        )

        print(
            "{} | {} | {}".format(
                record[
                    "dataset"
                ],
                record[
                    "role"
                ],
                record[
                    "relative_source_path"
                ],
            )
        )

        print(
            "  source frames: {} | "
            "size: {}x{} | "
            "r_fps: {} | avg_fps: {}"
            .format(
                source_probe[
                    "decoded_frame_count"
                ],
                source_probe[
                    "width"
                ],
                source_probe[
                    "height"
                ],
                source_probe[
                    "r_frame_rate"
                ],
                source_probe[
                    "avg_frame_rate"
                ],
            )
        )

        for condition in (
            "RSZ",
            "BLR",
            "H40",
            "PLT",
        ):
            spec = (
                EXPECTED_CONDITIONS[
                    condition
                ]
            )

            output_path = (
                processed_path(
                    record,
                    condition,
                )
            )

            if not output_path.is_file():
                raise FileNotFoundError(
                    output_path
                )

            if (
                output_path.stat().st_size
                <= 0
            ):
                raise RuntimeError(
                    "Empty output: {}".format(
                        output_path
                    )
                )

            output_probe = (
                probe_video(
                    output_path
                )
            )

            if (
                int(
                    output_probe[
                        "width"
                    ]
                )
                != int(
                    source_probe[
                        "width"
                    ]
                )
                or int(
                    output_probe[
                        "height"
                    ]
                )
                != int(
                    source_probe[
                        "height"
                    ]
                )
            ):
                raise RuntimeError(
                    "{} dimensions differ from "
                    "source: {}.".format(
                        condition,
                        output_path,
                    )
                )

            if (
                output_probe[
                    "decoded_frame_count"
                ]
                != source_probe[
                    "decoded_frame_count"
                ]
            ):
                raise RuntimeError(
                    "{} decoded-frame-count "
                    "mismatch: source={}, "
                    "processed={}."
                    .format(
                        condition,
                        source_probe[
                            "decoded_frame_count"
                        ],
                        output_probe[
                            "decoded_frame_count"
                        ],
                    )
                )

            if (
                output_probe[
                    "codec_name"
                ]
                != spec[
                    "codec"
                ]
            ):
                raise RuntimeError(
                    "{} codec mismatch: "
                    "{} != {}.".format(
                        condition,
                        output_probe[
                            "codec_name"
                        ],
                        spec[
                            "codec"
                        ],
                    )
                )

            if (
                output_probe[
                    "pix_fmt"
                ]
                != spec[
                    "pixel_format"
                ]
            ):
                raise RuntimeError(
                    "{} pixel-format mismatch: "
                    "{} != {}.".format(
                        condition,
                        output_probe[
                            "pix_fmt"
                        ],
                        spec[
                            "pixel_format"
                        ],
                    )
                )

            audio_count = (
                count_audio_streams(
                    output_path
                )
            )

            if audio_count != 0:
                raise RuntimeError(
                    "{} unexpectedly contains "
                    "{} audio stream(s)."
                    .format(
                        output_path,
                        audio_count,
                    )
                )

            fps_difference = (
                rate_difference(
                    source_probe[
                        "avg_frame_rate"
                    ],
                    output_probe[
                        "avg_frame_rate"
                    ],
                )
            )

            if (
                fps_difference
                > RATE_TOLERANCE_FPS
            ):
                raise RuntimeError(
                    "{} output FPS mismatch: "
                    "difference={:.9f}."
                    .format(
                        condition,
                        fps_difference,
                    )
                )

            pixel_audit = None

            if spec[
                "pixel_exact"
            ]:
                pixel_audit = (
                    pixel_exact_audit(
                        source_path=(
                            source_path
                        ),
                        processed_path=(
                            output_path
                        ),
                        condition=(
                            condition
                        ),
                        width=int(
                            source_probe[
                                "width"
                            ]
                        ),
                        height=int(
                            source_probe[
                                "height"
                            ]
                        ),
                    )
                )

                pixel_exact_checks += 1

            result = {
                "dataset": (
                    record[
                        "dataset"
                    ]
                ),
                "role": (
                    record[
                        "role"
                    ]
                ),
                "subgroup": (
                    record[
                        "subgroup"
                    ]
                ),
                "base_video_id": (
                    record[
                        "base_video_id"
                    ]
                ),
                "relative_source_path": (
                    record[
                        "relative_source_path"
                    ]
                ),
                "condition": (
                    condition
                ),
                "source_path": str(
                    source_path
                ),
                "processed_path": str(
                    output_path
                ),
                "source_decoded_frames": (
                    source_probe[
                        "decoded_frame_count"
                    ]
                ),
                "processed_decoded_frames": (
                    output_probe[
                        "decoded_frame_count"
                    ]
                ),
                "width": int(
                    output_probe[
                        "width"
                    ]
                ),
                "height": int(
                    output_probe[
                        "height"
                    ]
                ),
                "codec": (
                    output_probe[
                        "codec_name"
                    ]
                ),
                "pixel_format": (
                    output_probe[
                        "pix_fmt"
                    ]
                ),
                "source_r_frame_rate": (
                    source_probe[
                        "r_frame_rate"
                    ]
                ),
                "source_avg_frame_rate": (
                    source_probe[
                        "avg_frame_rate"
                    ]
                ),
                "processed_avg_frame_rate": (
                    output_probe[
                        "avg_frame_rate"
                    ]
                ),
                "rate_difference_fps": (
                    fps_difference
                ),
                "audio_stream_count": (
                    audio_count
                ),
                "size_bytes": (
                    output_path
                    .stat()
                    .st_size
                ),
                "pixel_exact_audit": (
                    pixel_audit
                ),
                "status": "passed",
            }

            results.append(
                result
            )

            extra = ""

            if pixel_audit is not None:
                extra = (
                    " | pixel max diff=0"
                )

            print(
                "  {:3s} | frames={} | "
                "{} | {} | {:.6f} fps{}"
                .format(
                    condition,
                    output_probe[
                        "decoded_frame_count"
                    ],
                    output_probe[
                        "codec_name"
                    ],
                    output_probe[
                        "pix_fmt"
                    ],
                    fps_difference,
                    extra,
                )
            )

        print()

    expected_results = (
        len(
            manifest
        )
        * len(
            EXPECTED_CONDITIONS
        )
    )

    if (
        len(
            results
        )
        != expected_results
    ):
        raise RuntimeError(
            "Expected {} audit records, "
            "got {}.".format(
                expected_results,
                len(
                    results
                ),
            )
        )

    summary = {
        "schema_version": 1,
        "manifest_record_count": (
            len(
                manifest
            )
        ),
        "condition_count": (
            len(
                EXPECTED_CONDITIONS
            )
        ),
        "output_record_count": (
            len(
                results
            )
        ),
        "pixel_exact_output_count": (
            pixel_exact_checks
        ),
        "expected_output_record_count": (
            expected_results
        ),
        "all_outputs_passed": True,
        "shared_processing_modules_imported": False,
        "rate_tolerance_fps": (
            RATE_TOLERANCE_FPS
        ),
        "records": results,
        "status": "passed",
    }

    AUDIT_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with AUDIT_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write(
            "\n"
        )

    print(
        "INDEPENDENT AUDIT SUMMARY"
    )
    print(
        "  smoke videos:          {}".format(
            len(
                manifest
            )
        )
    )
    print(
        "  processed outputs:     {}".format(
            len(
                results
            )
        )
    )
    print(
        "  metadata/frame checks: {}".format(
            len(
                results
            )
        )
    )
    print(
        "  pixel-exact checks:    {}".format(
            pixel_exact_checks
        )
    )
    print(
        "  pixel max difference:  0"
    )
    print(
        "  failures:              0"
    )
    print(
        "  artifact:              {}".format(
            AUDIT_OUTPUT_PATH
        )
    )
    print()
    print(
        "INDEPENDENT PROCESSING SMOKE AUDIT PASSED"
    )


if __name__ == "__main__":
    main()