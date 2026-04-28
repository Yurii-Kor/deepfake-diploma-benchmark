#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y"}


def _to_float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "null":
        return None
    return float(s)


def _to_int_or_none(value) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "null":
        return None
    return int(float(s))


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_label(values: List[int]) -> Optional[int]:
    if not values:
        return None
    return int(round(sum(values) / len(values)))


def _norm_key(value: str) -> str:
    return str(value).strip()


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, fieldnames: List[str], rows: List[Dict]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_frame_rows(
    frame_rows: List[Dict[str, str]],
    expected_frames: int,
) -> List[Dict]:
    grouped: Dict[Tuple[str, str, str, str, str, str, str, str, str, str], Dict] = {}

    for row in frame_rows:
        key = (
            row["run_id"],
            row["model_id"],
            row["family_id"],
            row["checkpoint_id"],
            row["dataset_id"],
            row["subset_id"],
            row["split"],
            row["condition_code"],
            row["video_id"],
            row["source_video_id"],
        )

        if key not in grouped:
            grouped[key] = {
                "run_id": row["run_id"],
                "model_id": row["model_id"],
                "family_id": row["family_id"],
                "checkpoint_id": row["checkpoint_id"],
                "dataset_id": row["dataset_id"],
                "subset_id": row["subset_id"],
                "split": row["split"],
                "condition_code": row["condition_code"],
                "video_id": row["video_id"],
                "source_video_id": row["source_video_id"],
                "scores": [],
                "labels": [],
                "score_available_flags": [],
                "face_found_flags": [],
                "failure_stage_values": [],
                "failure_reason_values": [],
            }

        g = grouped[key]

        score = _to_float_or_none(row.get("score"))
        label = _to_int_or_none(row.get("label"))

        if score is not None:
            g["scores"].append(score)
        if label is not None:
            g["labels"].append(label)

        g["score_available_flags"].append(_to_bool(row.get("score_available", True)))
        g["face_found_flags"].append(_to_bool(row.get("face_found", True)))

        stage = str(row.get("failure_stage", "")).strip()
        reason = str(row.get("failure_reason", "")).strip()
        if stage:
            g["failure_stage_values"].append(stage)
        if reason:
            g["failure_reason_values"].append(reason)

    video_rows = []
    for _, g in grouped.items():
        n_frames_scored = len(g["scores"])
        n_frames_expected = expected_frames
        n_frames_missing = max(n_frames_expected - n_frames_scored, 0)
        frame_score_coverage = (
            n_frames_scored / n_frames_expected if n_frames_expected > 0 else None
        )

        score = _safe_mean(g["scores"])
        label = _safe_label(g["labels"])

        score_available = n_frames_scored > 0
        face_found = n_frames_scored > 0
        partial_score = score_available and n_frames_scored < n_frames_expected

        failure_stage = ""
        failure_reason = ""
        if not score_available:
            failure_stage = "video_aggregation"
            failure_reason = "no_scored_frames_for_video"
        elif partial_score:
            failure_stage = "video_aggregation"
            failure_reason = "partial_frame_coverage"

        video_rows.append({
            "run_id": g["run_id"],
            "model_id": g["model_id"],
            "family_id": g["family_id"],
            "checkpoint_id": g["checkpoint_id"],
            "dataset_id": g["dataset_id"],
            "subset_id": g["subset_id"],
            "split": g["split"],
            "condition_code": g["condition_code"],
            "video_id": g["video_id"],
            "source_video_id": g["source_video_id"],
            "score": score,
            "label": label,
            "n_frames_expected": n_frames_expected,
            "n_frames_scored": n_frames_scored,
            "n_frames_missing": n_frames_missing,
            "frame_score_coverage": frame_score_coverage,
            "partial_score": partial_score,
            "score_available": score_available,
            "face_found": face_found,
            "failure_stage": failure_stage,
            "failure_reason": failure_reason,
        })

    return video_rows


def load_expected_videos(expected_csv: str, expected_key: str) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(expected_csv)
    index = {}
    for row in rows:
        if expected_key not in row:
            raise ValueError(f"Expected file is missing key column: {expected_key}")
        index[_norm_key(row[expected_key])] = row
    return index


def attach_no_score_rows(
    video_rows: List[Dict],
    expected_index: Dict[str, Dict[str, str]],
    expected_key: str,
    expected_frames: int,
) -> Tuple[List[Dict], List[Dict]]:
    observed_index = {_norm_key(r[expected_key]): r for r in video_rows}
    merged_rows = list(video_rows)
    exclusion_rows = []

    for key, exp in expected_index.items():
        if key in observed_index:
            continue

        row = {
            "run_id": exp.get("run_id", ""),
            "model_id": exp.get("model_id", ""),
            "family_id": exp.get("family_id", ""),
            "checkpoint_id": exp.get("checkpoint_id", ""),
            "dataset_id": exp.get("dataset_id", ""),
            "subset_id": exp.get("subset_id", ""),
            "split": exp.get("split", ""),
            "condition_code": exp.get("condition_code", ""),
            "video_id": exp.get("video_id", ""),
            "source_video_id": exp.get("source_video_id", exp.get("video_id", "")),
            "score": None,
            "label": _to_int_or_none(exp.get("label")),
            "n_frames_expected": expected_frames,
            "n_frames_scored": 0,
            "n_frames_missing": expected_frames,
            "frame_score_coverage": 0.0,
            "partial_score": False,
            "score_available": False,
            "face_found": False,
            "failure_stage": "video_aggregation",
            "failure_reason": "no_scored_frames_for_video",
        }
        merged_rows.append(row)

        exclusion_rows.append({
            "run_id": row["run_id"],
            "model_id": row["model_id"],
            "family_id": row["family_id"],
            "checkpoint_id": row["checkpoint_id"],
            "dataset_id": row["dataset_id"],
            "subset_id": row["subset_id"],
            "split": row["split"],
            "condition_code": row["condition_code"],
            "video_id": row["video_id"],
            "source_video_id": row["source_video_id"],
            "label": row["label"],
            "stage": row["failure_stage"],
            "reason_code": row["failure_reason"],
            "details": "Present in expected video universe, absent from frame-level scored rows",
        })

    return merged_rows, exclusion_rows


def build_exclusion_rows_from_partials(video_rows: List[Dict]) -> List[Dict]:
    rows = []
    for r in video_rows:
        if not r["score_available"]:
            rows.append({
                "run_id": r["run_id"],
                "model_id": r["model_id"],
                "family_id": r["family_id"],
                "checkpoint_id": r["checkpoint_id"],
                "dataset_id": r["dataset_id"],
                "subset_id": r["subset_id"],
                "split": r["split"],
                "condition_code": r["condition_code"],
                "video_id": r["video_id"],
                "source_video_id": r["source_video_id"],
                "label": r["label"],
                "stage": r["failure_stage"],
                "reason_code": r["failure_reason"],
                "details": "No scored frames after aggregation",
            })
        elif r["partial_score"]:
            rows.append({
                "run_id": r["run_id"],
                "model_id": r["model_id"],
                "family_id": r["family_id"],
                "checkpoint_id": r["checkpoint_id"],
                "dataset_id": r["dataset_id"],
                "subset_id": r["subset_id"],
                "split": r["split"],
                "condition_code": r["condition_code"],
                "video_id": r["video_id"],
                "source_video_id": r["source_video_id"],
                "label": r["label"],
                "stage": r["failure_stage"],
                "reason_code": r["failure_reason"],
                "details": f"Scored {r['n_frames_scored']} of {r['n_frames_expected']} expected frames",
            })
    return rows


def summarize_coverage(video_rows: List[Dict]) -> Dict[str, float]:
    n_total = len(video_rows)
    n_scored = sum(1 for r in video_rows if r["score_available"])
    n_face_found = sum(1 for r in video_rows if r["face_found"])
    n_partial = sum(1 for r in video_rows if r["partial_score"])
    n_no_score = sum(1 for r in video_rows if not r["score_available"])

    return {
        "n_total_videos": n_total,
        "n_scored_videos": n_scored,
        "n_face_found_videos": n_face_found,
        "n_partial_videos": n_partial,
        "n_no_score_videos": n_no_score,
        "score_available_rate": (n_scored / n_total) if n_total else math.nan,
        "face_found_rate": (n_face_found / n_total) if n_total else math.nan,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, help="Frame/sample-level raw_scores CSV")
    parser.add_argument("--output-csv", required=True, help="Video-level aggregated CSV")
    parser.add_argument("--exclusion-log", required=True, help="Output exclusion log CSV")
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=8,
        help="Expected number of scored frames per video for this run",
    )
    parser.add_argument(
        "--expected-videos-csv",
        default=None,
        help="Optional expected video universe CSV to materialize no-score videos",
    )
    parser.add_argument(
        "--expected-key",
        choices=["video_id", "source_video_id"],
        default="video_id",
        help="Join key for expected videos CSV",
    )
    args = parser.parse_args()

    frame_rows = read_csv_rows(args.input_csv)
    if not frame_rows:
        raise ValueError("Input CSV is empty")

    video_rows = aggregate_frame_rows(
        frame_rows=frame_rows,
        expected_frames=args.expected_frames,
    )

    exclusion_rows = build_exclusion_rows_from_partials(video_rows)

    if args.expected_videos_csv:
        expected_index = load_expected_videos(args.expected_videos_csv, args.expected_key)
        video_rows, extra_exclusions = attach_no_score_rows(
            video_rows=video_rows,
            expected_index=expected_index,
            expected_key=args.expected_key,
            expected_frames=args.expected_frames,
        )
        exclusion_rows.extend(extra_exclusions)

    video_rows.sort(key=lambda r: (
        r["run_id"], r["dataset_id"], r["subset_id"], r["condition_code"], r["video_id"]
    ))
    exclusion_rows.sort(key=lambda r: (
        r["run_id"], r["dataset_id"], r["subset_id"], r["condition_code"], r["video_id"], r["reason_code"]
    ))

    video_fieldnames = [
        "run_id",
        "model_id",
        "family_id",
        "checkpoint_id",
        "dataset_id",
        "subset_id",
        "split",
        "condition_code",
        "video_id",
        "source_video_id",
        "score",
        "label",
        "n_frames_expected",
        "n_frames_scored",
        "n_frames_missing",
        "frame_score_coverage",
        "partial_score",
        "score_available",
        "face_found",
        "failure_stage",
        "failure_reason",
    ]
    exclusion_fieldnames = [
        "run_id",
        "model_id",
        "family_id",
        "checkpoint_id",
        "dataset_id",
        "subset_id",
        "split",
        "condition_code",
        "video_id",
        "source_video_id",
        "label",
        "stage",
        "reason_code",
        "details",
    ]

    write_csv(args.output_csv, video_fieldnames, video_rows)
    write_csv(args.exclusion_log, exclusion_fieldnames, exclusion_rows)

    summary = summarize_coverage(video_rows)
    print("Video-level aggregation done.")
    print(f"Input frame rows: {len(frame_rows)}")
    print(f"Output video rows: {len(video_rows)}")
    print(f"Exclusion rows: {len(exclusion_rows)}")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()