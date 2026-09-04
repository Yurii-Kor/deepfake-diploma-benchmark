from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from processing_common import (
    RawVideoReader,
    apply_operations,
    load_config,
    parse_rate,
    probe_video,
    rates_equal,
    select_nominal_fps,
)


def rate_difference(
    first: str,
    second: str,
) -> float | None:
    first_rate = parse_rate(first)
    second_rate = parse_rate(second)

    if (
        first_rate is None
        or second_rate is None
    ):
        return None

    return abs(
        float(first_rate)
        - float(second_rate)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one processed study video "
            "against its clean source."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--condition",
        required=True,
    )

    parser.add_argument(
        "--source",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--processed",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    if args.condition not in config["conditions"]:
        raise ValueError(
            f"Unknown condition '{args.condition}'. "
            f"Available conditions: "
            f"{sorted(config['conditions'])}"
        )

    if not args.source.is_file():
        raise FileNotFoundError(
            f"Source video does not exist: "
            f"{args.source}"
        )

    if not args.processed.is_file():
        raise FileNotFoundError(
            f"Processed video does not exist: "
            f"{args.processed}"
        )

    ffmpeg_bin = config["tools"]["ffmpeg"]
    ffprobe_bin = config["tools"]["ffprobe"]

    condition_config = config["conditions"][
        args.condition
    ]

    source_probe = probe_video(
        args.source,
        ffprobe_bin=ffprobe_bin,
    )

    processed_probe = probe_video(
        args.processed,
        ffprobe_bin=ffprobe_bin,
    )

    if (
        source_probe["width"]
        != processed_probe["width"]
    ):
        raise RuntimeError(
            "Width mismatch: "
            f"{source_probe['width']} != "
            f"{processed_probe['width']}"
        )

    if (
        source_probe["height"]
        != processed_probe["height"]
    ):
        raise RuntimeError(
            "Height mismatch: "
            f"{source_probe['height']} != "
            f"{processed_probe['height']}"
        )

    encoder_config = condition_config[
        "encoder"
    ]

    expected_codec = encoder_config.get(
        "expected_codec_name",
        encoder_config["codec"],
    )

    if (
        processed_probe["codec_name"]
        != expected_codec
    ):
        raise RuntimeError(
            "Codec mismatch: "
            f"expected={expected_codec}, "
            f"actual="
            f"{processed_probe['codec_name']}"
        )

    expected_pixel_format = encoder_config[
        "pixel_format"
    ]

    if (
        processed_probe["pixel_format"]
        != expected_pixel_format
    ):
        raise RuntimeError(
            "Pixel-format mismatch: "
            f"expected={expected_pixel_format}, "
            f"actual="
            f"{processed_probe['pixel_format']}"
        )

    require_pixel_exact = (
        condition_config["validation"].get(
            "require_pixel_exact_roundtrip",
            False,
        )
    )

    width = source_probe["width"]
    height = source_probe["height"]

    frame_count = 0
    max_absolute_difference = 0
    first_pixel_mismatch = None

    with RawVideoReader(
        args.source,
        width=width,
        height=height,
        ffmpeg_bin=ffmpeg_bin,
    ) as source_reader, RawVideoReader(
        args.processed,
        width=width,
        height=height,
        ffmpeg_bin=ffmpeg_bin,
    ) as processed_reader:
        while True:
            source_frame = (
                source_reader.read_frame()
            )

            processed_frame = (
                processed_reader.read_frame()
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
                    "Decoded frame-count mismatch "
                    "between source and processed video."
                )

            if require_pixel_exact:
                expected_frame = apply_operations(
                    source_frame,
                    condition_config["operations"],
                )

                difference = np.abs(
                    expected_frame.astype(
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

                if (
                    current_max != 0
                    and first_pixel_mismatch is None
                ):
                    first_pixel_mismatch = (
                        frame_count
                    )

            frame_count += 1

    if frame_count == 0:
        raise RuntimeError(
            "No frames were decoded "
            "during validation."
        )

    if (
        require_pixel_exact
        and max_absolute_difference != 0
    ):
        raise RuntimeError(
            "Lossless round-trip validation failed.\n"
            f"First mismatching frame: "
            f"{first_pixel_mismatch}\n"
            "Maximum absolute pixel difference: "
            f"{max_absolute_difference}"
        )

    target_fps = select_nominal_fps(
        source_probe
    )

    processed_fps = processed_probe[
        "avg_frame_rate"
    ]

    rate_tolerance = float(
        config["video_policy"].get(
            "output_rate_tolerance_fps",
            0.001,
        )
    )

    if not rates_equal(
        target_fps,
        processed_fps,
        tolerance=rate_tolerance,
    ):
        difference = rate_difference(
            target_fps,
            processed_fps,
        )

        raise RuntimeError(
            "Processed average frame rate "
            "does not match the configured "
            "processing rate.\n"
            f"target={target_fps}\n"
            f"processed={processed_fps}\n"
            f"difference={difference}\n"
            f"tolerance={rate_tolerance}"
        )

    difference = rate_difference(
        target_fps,
        processed_fps,
    )

    print("VALIDATION PASSED")
    print(
        f"Condition:          "
        f"{args.condition}"
    )
    print(
        f"Source:             "
        f"{args.source}"
    )
    print(
        f"Processed:          "
        f"{args.processed}"
    )
    print(
        f"Decoded frames:     "
        f"{frame_count}"
    )
    print(
        f"Dimensions:         "
        f"{width}x{height}"
    )

    print(
        f"Source r FPS:       "
        f"{source_probe['r_frame_rate']}"
    )
    print(
        f"Source avg FPS:     "
        f"{source_probe['avg_frame_rate']}"
    )
    print(
        f"Processing FPS:     "
        f"{target_fps}"
    )
    print(
        f"Processed r FPS:    "
        f"{processed_probe['r_frame_rate']}"
    )
    print(
        f"Processed avg FPS:  "
        f"{processed_fps}"
    )
    print(
        f"Rate difference:    "
        f"{difference:.9f} fps"
    )

    print(
        f"Codec:              "
        f"{processed_probe['codec_name']}"
    )
    print(
        f"Pixel format:       "
        f"{processed_probe['pixel_format']}"
    )

    if require_pixel_exact:
        print(
            "Pixel round-trip:   exact"
        )
        print(
            "Max pixel diff:      0"
        )
    else:
        print(
            "Pixel round-trip:   "
            "not required for lossy condition"
        )


if __name__ == "__main__":
    main()