from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from processing_common import (
    RawVideoReader,
    apply_operations,
    load_config,
    probe_video,
    rates_equal,
    select_nominal_fps,
)


def build_frame_encoder_command(
    ffmpeg_bin: str,
    output_path: Path,
    width: int,
    height: int,
    fps: str,
    encoder_config: dict,
) -> list[str]:
    command = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        fps,
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-an",
    ]

    codec = encoder_config["codec"]

    if codec == "ffv1":
        command.extend([
            "-c:v",
            "ffv1",
            "-level",
            str(encoder_config.get("level", 3)),
            "-pix_fmt",
            encoder_config["pixel_format"],
        ])

    elif codec == "libx264":
        command.extend([
            "-c:v",
            "libx264",
            "-preset",
            encoder_config["preset"],
            "-crf",
            str(encoder_config["crf"]),
            "-pix_fmt",
            encoder_config["pixel_format"],
        ])

        movflags = encoder_config.get("movflags")

        if movflags:
            command.extend([
                "-movflags",
                movflags,
            ])

    else:
        raise ValueError(
            f"Unsupported encoder codec: {codec}"
        )

    command.append(str(output_path))

    return command


def build_direct_transcode_command(
    ffmpeg_bin: str,
    input_path: Path,
    output_path: Path,
    fps: str,
    encoder_config: dict,
) -> list[str]:
    if encoder_config["codec"] != "libx264":
        raise ValueError(
            "The current direct-transcode path supports libx264 only."
        )

    # Input -r deliberately ignores source container timestamps and
    # reconstructs a deterministic temporal grid from decoded frame
    # index using the clean source average frame rate.
    #
    # This preserves one decoded input frame -> one encoded output
    # frame while avoiding condition-specific PTS irregularities.
    command = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",

        # IMPORTANT: -r is an input option here and must precede -i.
        "-r",
        fps,
        "-i",
        str(input_path),

        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        encoder_config["preset"],
        "-crf",
        str(encoder_config["crf"]),
        "-pix_fmt",
        encoder_config["pixel_format"],
        "-fps_mode",
        "passthrough",
    ]

    movflags = encoder_config.get("movflags")

    if movflags:
        command.extend([
            "-movflags",
            movflags,
        ])

    command.append(str(output_path))

    return command


def run_direct_transcode(
    ffmpeg_bin: str,
    input_path: Path,
    temp_output: Path,
    fps: str,
    encoder_config: dict,
) -> None:
    command = build_direct_transcode_command(
        ffmpeg_bin=ffmpeg_bin,
        input_path=input_path,
        output_path=temp_output,
        fps=fps,
        encoder_config=encoder_config,
    )

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg direct transcode failed:\n"
            f"{result.stderr.decode(errors='replace').strip()}"
        )


def run_frame_processing(
    ffmpeg_bin: str,
    input_path: Path,
    temp_output: Path,
    width: int,
    height: int,
    fps: str,
    condition_config: dict,
) -> int:
    encoder_config = condition_config["encoder"]

    encoder_command = build_frame_encoder_command(
        ffmpeg_bin=ffmpeg_bin,
        output_path=temp_output,
        width=width,
        height=height,
        fps=fps,
        encoder_config=encoder_config,
    )

    encoder = subprocess.Popen(
        encoder_command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if encoder.stdin is None or encoder.stderr is None:
        raise RuntimeError(
            "Unable to create FFmpeg encoder pipes."
        )

    frame_count = 0

    try:
        with RawVideoReader(
            input_path,
            width=width,
            height=height,
            ffmpeg_bin=ffmpeg_bin,
        ) as reader:
            while True:
                frame = reader.read_frame()

                if frame is None:
                    break

                processed = apply_operations(
                    frame,
                    condition_config["operations"],
                )

                if processed.shape != frame.shape:
                    raise RuntimeError(
                        "Processing changed final frame geometry: "
                        f"{frame.shape} -> {processed.shape}"
                    )

                encoder.stdin.write(
                    processed.tobytes()
                )

                frame_count += 1

        encoder.stdin.close()

        return_code = encoder.wait()

        stderr = encoder.stderr.read()
        encoder.stderr.close()

        if return_code != 0:
            raise RuntimeError(
                "FFmpeg encoding failed:\n"
                f"{stderr.decode(errors='replace').strip()}"
            )

    except Exception:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()

        if (
            encoder.stdin is not None
            and not encoder.stdin.closed
        ):
            encoder.stdin.close()

        if (
            encoder.stderr is not None
            and not encoder.stderr.closed
        ):
            encoder.stderr.close()

        raise

    if frame_count == 0:
        raise RuntimeError(
            "No frames were decoded from the source video."
        )

    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one study processing condition "
            "for one video."
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
        "--input",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.condition not in config["conditions"]:
        raise ValueError(
            f"Unknown condition '{args.condition}'. "
            f"Available conditions: "
            f"{sorted(config['conditions'])}"
        )

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Input video does not exist: {args.input}"
        )

    condition_config = config["conditions"][
        args.condition
    ]

    expected_extension = condition_config[
        "output_extension"
    ]

    if (
        args.output.suffix.lower()
        != expected_extension.lower()
    ):
        raise ValueError(
            f"{args.condition} output must use extension "
            f"{expected_extension}, "
            f"got {args.output.suffix}."
        )

    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"Output already exists: {args.output}\n"
            "Use --force only if you intentionally "
            "want to replace it."
        )

    ffmpeg_bin = config["tools"]["ffmpeg"]
    ffprobe_bin = config["tools"]["ffprobe"]

    source_probe = probe_video(
        args.input,
        ffprobe_bin=ffprobe_bin,
    )

    width = source_probe["width"]
    height = source_probe["height"]

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid source dimensions: "
            f"{width}x{height}"
        )

    if config["video_policy"].get(
        "require_cfr_rate_match",
        False,
    ):
        avg_rate = source_probe.get(
            "avg_frame_rate"
        )

        r_rate = source_probe.get(
            "r_frame_rate"
        )

        if (
            avg_rate
            and r_rate
            and not rates_equal(
                avg_rate,
                r_rate,
            )
        ):
            raise ValueError(
                "Source video does not satisfy "
                "the configured CFR preflight rule:\n"
                f"avg_frame_rate={avg_rate}, "
                f"r_frame_rate={r_rate}"
            )

    fps = select_nominal_fps(
        source_probe
    )

    encoder_config = condition_config[
        "encoder"
    ]

    pixel_format = encoder_config[
        "pixel_format"
    ]

    if (
        pixel_format == "yuv420p"
        and (
            width % 2 != 0
            or height % 2 != 0
        )
    ):
        raise ValueError(
            "yuv420p output requires even "
            "source dimensions, "
            f"got {width}x{height}."
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_output = args.output.with_name(
        f"{args.output.stem}.tmp"
        f"{args.output.suffix}"
    )

    if temp_output.exists():
        temp_output.unlink()

    pipeline = condition_config.get(
        "pipeline",
        "frame_processing",
    )

    frame_count = None

    try:
        if pipeline == "direct_transcode":
            run_direct_transcode(
                ffmpeg_bin=ffmpeg_bin,
                input_path=args.input,
                temp_output=temp_output,
                fps=fps,
                encoder_config=encoder_config,
            )

        elif pipeline == "frame_processing":
            frame_count = run_frame_processing(
                ffmpeg_bin=ffmpeg_bin,
                input_path=args.input,
                temp_output=temp_output,
                width=width,
                height=height,
                fps=fps,
                condition_config=condition_config,
            )

        else:
            raise ValueError(
                "Unsupported processing pipeline: "
                f"{pipeline}"
            )

        if (
            not temp_output.is_file()
            or temp_output.stat().st_size == 0
        ):
            raise RuntimeError(
                "Processing did not produce "
                "a valid temporary output: "
                f"{temp_output}"
            )

        os.replace(
            temp_output,
            args.output,
        )

    except Exception:
        if temp_output.exists():
            temp_output.unlink()

        raise

    output_probe = probe_video(
        args.output,
        ffprobe_bin=ffprobe_bin,
    )

    print(
        "Processing completed successfully."
    )
    print(
        f"Condition:       {args.condition}"
    )
    print(
        f"Pipeline:        {pipeline}"
    )
    print(
        f"Input:           {args.input}"
    )
    print(
        f"Output:          {args.output}"
    )

    if frame_count is not None:
        print(
            f"Frames:          {frame_count}"
        )
    else:
        print(
            "Frames:          verified by validator"
        )

    print(
        f"Dimensions:      {width}x{height}"
    )
    print(
        f"Processing FPS:  {fps}"
    )
    print(
        f"Encoder:         "
        f"{encoder_config['codec']}"
    )
    print(
        f"Output codec:    "
        f"{output_probe['codec_name']}"
    )
    print(
        f"Pixel format:    "
        f"{output_probe['pixel_format']}"
    )


if __name__ == "__main__":
    main()