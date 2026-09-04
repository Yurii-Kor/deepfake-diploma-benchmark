from __future__ import annotations

import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration file: {config_path}")

    if "tools" not in config:
        raise ValueError("Missing 'tools' section in processing configuration.")

    if "conditions" not in config:
        raise ValueError("Missing 'conditions' section in processing configuration.")

    return config


def parse_rate(value: str | None) -> Fraction | None:
    if not value:
        return None

    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None

    if rate <= 0:
        return None

    return rate


def rates_equal(
    first: str | None,
    second: str | None,
    tolerance: float = 1e-9,
) -> bool:
    first_rate = parse_rate(first)
    second_rate = parse_rate(second)

    if first_rate is None or second_rate is None:
        return False

    return abs(float(first_rate) - float(second_rate)) <= tolerance


def select_nominal_fps(probe: dict[str, Any]) -> str:
    avg_frame_rate = probe.get("avg_frame_rate")
    r_frame_rate = probe.get("r_frame_rate")

    if parse_rate(avg_frame_rate) is not None:
        return avg_frame_rate

    if parse_rate(r_frame_rate) is not None:
        return r_frame_rate

    raise ValueError("Unable to determine a valid source frame rate.")


def probe_video(
    path: Path,
    ffprobe_bin: str = "ffprobe",
) -> dict[str, Any]:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,pix_fmt,width,height,"
            "r_frame_rate,avg_frame_rate,nb_frames:"
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
        raise RuntimeError(
            f"ffprobe failed for {path}:\n{result.stderr.strip()}"
        )

    data = json.loads(result.stdout)
    streams = data.get("streams", [])

    if not streams:
        raise ValueError(f"No video stream found in: {path}")

    stream = streams[0]
    format_info = data.get("format", {})

    return {
        "codec_name": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "nb_frames": stream.get("nb_frames"),
        "duration": format_info.get("duration"),
    }


def _opencv_interpolation(name: str) -> int:
    mapping = {
        "INTER_CUBIC": cv2.INTER_CUBIC,
        "INTER_LINEAR": cv2.INTER_LINEAR,
        "INTER_AREA": cv2.INTER_AREA,
        "INTER_NEAREST": cv2.INTER_NEAREST,
    }

    if name not in mapping:
        raise ValueError(f"Unsupported interpolation mode: {name}")

    return mapping[name]


def _opencv_border_type(name: str) -> int:
    mapping = {
        "BORDER_REFLECT": cv2.BORDER_REFLECT,
        "BORDER_REFLECT_101": cv2.BORDER_REFLECT_101,
        "BORDER_REPLICATE": cv2.BORDER_REPLICATE,
    }

    if name not in mapping:
        raise ValueError(f"Unsupported border type: {name}")

    return mapping[name]


def _scaled_dimension(size: int, scale: float) -> int:
    scaled = int(math.floor(size * scale + 0.5))
    return max(1, scaled)


def resize_restore(
    frame: np.ndarray,
    scale: float,
    interpolation_name: str,
) -> np.ndarray:
    height, width = frame.shape[:2]

    scaled_width = _scaled_dimension(width, scale)
    scaled_height = _scaled_dimension(height, scale)

    interpolation = _opencv_interpolation(interpolation_name)

    downscaled = cv2.resize(
        frame,
        (scaled_width, scaled_height),
        interpolation=interpolation,
    )

    restored = cv2.resize(
        downscaled,
        (width, height),
        interpolation=interpolation,
    )

    return restored


def gaussian_blur(
    frame: np.ndarray,
    kernel_size: int,
    sigma_x: float,
    sigma_y: float,
    border_type_name: str,
) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"Gaussian kernel size must be a positive odd number, got {kernel_size}."
        )

    return cv2.GaussianBlur(
        frame,
        (kernel_size, kernel_size),
        sigmaX=sigma_x,
        sigmaY=sigma_y,
        borderType=_opencv_border_type(border_type_name),
    )


def apply_operations(
    frame: np.ndarray,
    operations: list[dict[str, Any]],
) -> np.ndarray:
    result = frame

    for operation in operations:
        operation_type = operation["type"]

        if operation_type == "resize_restore":
            result = resize_restore(
                result,
                scale=float(operation["scale"]),
                interpolation_name=operation["interpolation"],
            )

        elif operation_type == "gaussian_blur":
            result = gaussian_blur(
                result,
                kernel_size=int(operation["kernel_size"]),
                sigma_x=float(operation["sigma_x"]),
                sigma_y=float(operation["sigma_y"]),
                border_type_name=operation["border_type"],
            )

        else:
            raise ValueError(
                f"Unsupported processing operation: {operation_type}"
            )

    return np.ascontiguousarray(result)


def _read_exact(stream, byte_count: int) -> bytes:
    chunks = []
    remaining = byte_count

    while remaining > 0:
        chunk = stream.read(remaining)

        if not chunk:
            break

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


class RawVideoReader:
    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        ffmpeg_bin: str = "ffmpeg",
    ):
        self.path = path
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        self.finished = False

        command = [
            ffmpeg_bin,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if self.process.stdout is None or self.process.stderr is None:
            raise RuntimeError("Unable to create FFmpeg decoding pipes.")

    def read_frame(self) -> np.ndarray | None:
        data = _read_exact(self.process.stdout, self.frame_bytes)

        if len(data) == 0:
            return None

        if len(data) != self.frame_bytes:
            raise RuntimeError(
                f"Partial decoded frame from {self.path}: "
                f"expected {self.frame_bytes} bytes, got {len(data)}."
            )

        frame = np.frombuffer(data, dtype=np.uint8)
        frame = frame.reshape((self.height, self.width, 3))

        return frame.copy()

    def finish(self) -> None:
        if self.finished:
            return

        self.finished = True

        if self.process.stdout is not None:
            self.process.stdout.close()

        return_code = self.process.wait()

        stderr = b""
        if self.process.stderr is not None:
            stderr = self.process.stderr.read()
            self.process.stderr.close()

        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg decoding failed for {self.path}:\n"
                f"{stderr.decode(errors='replace').strip()}"
            )

    def abort(self) -> None:
        if self.finished:
            return

        self.finished = True

        if self.process.poll() is None:
            self.process.kill()

        self.process.wait()

        if self.process.stdout is not None:
            self.process.stdout.close()

        if self.process.stderr is not None:
            self.process.stderr.close()

    def __enter__(self) -> "RawVideoReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.abort()