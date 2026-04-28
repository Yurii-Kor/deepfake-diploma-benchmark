#!/usr/bin/env python3
import argparse
import csv
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple


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


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, fieldnames: List[str], rows: List[Dict]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_pooled_run_id(model_id: str, condition_code: str, checkpoint_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_id))[0]
    return f"{model_id}_{condition_code}_{ckpt_stem}_FFPP_POOLED_VAL_{ts}"


def validate_common_metadata(rows_by_file: List[Tuple[str, List[Dict[str, str]]]]):
    common_keys = ["model_id", "family_id", "checkpoint_id", "dataset_id", "split", "condition_code"]
    expected = None

    for path, rows in rows_by_file:
        if not rows:
            raise ValueError(f"Input CSV is empty: {path}")
        first = rows[0]
        current = tuple(first.get(k, "") for k in common_keys)

        if expected is None:
            expected = current
        elif current != expected:
            raise ValueError(
                f"Metadata mismatch across input files.\n"
                f"Expected {common_keys}={expected}\n"
                f"Got      {common_keys}={current}\n"
                f"File: {path}"
            )

    return {
        k: rows_by_file[0][1][0].get(k, "")
        for k in common_keys
    }


def preference_key(
    row: Dict[str, str],
    subset_priority: Dict[str, int],
) -> Tuple:
    score_available = int(_to_bool(row.get("score_available", True)))
    face_found = int(_to_bool(row.get("face_found", False)))
    frame_score_coverage = _to_float_or_none(row.get("frame_score_coverage"))
    n_frames_scored = _to_int_or_none(row.get("n_frames_scored"))

    if frame_score_coverage is None or math.isnan(frame_score_coverage):
        frame_score_coverage = -1.0
    if n_frames_scored is None:
        n_frames_scored = -1

    source_subset_id = row.get("source_subset_id", row.get("subset_id", ""))
    return (
        -score_available,
        -face_found,
        -frame_score_coverage,
        -n_frames_scored,
        subset_priority.get(source_subset_id, 999),
        row.get("original_run_id", row.get("run_id", "")),
        row.get("video_id", ""),
    )


def normalize_rows(
    rows_by_file: List[Tuple[str, List[Dict[str, str]]]],
    pooled_run_id: str,
    pooled_subset_id: str,
) -> List[Dict]:
    normalized = []

    for path, rows in rows_by_file:
        for r in rows:
            out = dict(r)
            out["original_run_id"] = r.get("run_id", "")
            out["source_subset_id"] = r.get("subset_id", "")
            out["source_csv"] = path

            # Normalize pooled identity for downstream threshold computation
            out["run_id"] = pooled_run_id
            out["subset_id"] = pooled_subset_id

            normalized.append(out)

    return normalized


def dedupe_real_rows(
    normalized_rows: List[Dict[str, str]],
    subset_order: List[str],
) -> Tuple[List[Dict], List[Dict]]:
    subset_priority = {subset: i for i, subset in enumerate(subset_order)}

    fake_rows = [r for r in normalized_rows if _to_int_or_none(r.get("label")) == 1]
    real_rows = [r for r in normalized_rows if _to_int_or_none(r.get("label")) == 0]

    groups: Dict[str, List[Dict]] = {}
    for r in real_rows:
        key = r.get("source_video_id", "").strip()
        if not key:
            key = r.get("video_id", "").strip()
        groups.setdefault(key, []).append(r)

    kept_real_rows = []
    dedup_log_rows = []

    for source_video_id, candidates in groups.items():
        ordered = sorted(candidates, key=lambda r: preference_key(r, subset_priority))
        kept = ordered[0]

        kept_real_rows.append(kept)

        if len(candidates) > 1:
            for idx, cand in enumerate(ordered):
                dedup_log_rows.append({
                    "pooled_run_id": kept.get("run_id", ""),
                    "source_video_id": source_video_id,
                    "candidate_video_id": cand.get("video_id", ""),
                    "candidate_subset_id": cand.get("source_subset_id", ""),
                    "candidate_original_run_id": cand.get("original_run_id", ""),
                    "candidate_score": cand.get("score", ""),
                    "candidate_label": cand.get("label", ""),
                    "candidate_score_available": cand.get("score_available", ""),
                    "candidate_face_found": cand.get("face_found", ""),
                    "candidate_frame_score_coverage": cand.get("frame_score_coverage", ""),
                    "candidate_n_frames_scored": cand.get("n_frames_scored", ""),
                    "is_kept": idx == 0,
                    "selection_reason": "best_real_duplicate_by_coverage_then_frames_then_subset_priority",
                    "kept_subset_id": kept.get("source_subset_id", ""),
                    "kept_original_run_id": kept.get("original_run_id", ""),
                    "dedup_group_size": len(candidates),
                })

    pooled_rows = fake_rows + kept_real_rows
    pooled_rows.sort(key=lambda r: (
        r.get("run_id", ""),
        r.get("dataset_id", ""),
        r.get("subset_id", ""),
        r.get("condition_code", ""),
        _to_int_or_none(r.get("label")) if _to_int_or_none(r.get("label")) is not None else 99,
        r.get("source_subset_id", ""),
        r.get("video_id", ""),
    ))

    return pooled_rows, dedup_log_rows


def summarize_rows(rows: List[Dict[str, str]]) -> Dict[str, int]:
    n_total = len(rows)
    n_real = sum(1 for r in rows if _to_int_or_none(r.get("label")) == 0)
    n_fake = sum(1 for r in rows if _to_int_or_none(r.get("label")) == 1)
    n_scored = sum(1 for r in rows if _to_bool(r.get("score_available", True)) and _to_float_or_none(r.get("score")) is not None)
    n_partial = sum(1 for r in rows if _to_bool(r.get("partial_score", False)))
    return {
        "n_total_rows": n_total,
        "n_real_rows": n_real,
        "n_fake_rows": n_fake,
        "n_scored_rows": n_scored,
        "n_partial_rows": n_partial,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csvs",
        nargs="+",
        required=True,
        help="Validation video CSVs for FF-DF / FF-F2F / FF-FS / FF-NT",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Pooled calibration CSV path",
    )
    parser.add_argument(
        "--dedup-log",
        required=True,
        help="Dedup log CSV path",
    )
    parser.add_argument(
        "--pooled-subset-id",
        default="FFPP-POOLED-CALIBRATION",
        help="subset_id to assign to pooled rows",
    )
    parser.add_argument(
        "--subset-order",
        nargs="+",
        default=["FF-DF", "FF-F2F", "FF-FS", "FF-NT"],
        help="Preference order used only when resolving duplicate real rows",
    )
    args = parser.parse_args()

    rows_by_file = []
    for path in args.input_csvs:
        rows = read_csv_rows(path)
        rows_by_file.append((path, rows))

    common_meta = validate_common_metadata(rows_by_file)
    pooled_run_id = build_pooled_run_id(
        model_id=common_meta["model_id"],
        condition_code=common_meta["condition_code"],
        checkpoint_id=common_meta["checkpoint_id"],
    )

    normalized_rows = normalize_rows(
        rows_by_file=rows_by_file,
        pooled_run_id=pooled_run_id,
        pooled_subset_id=args.pooled_subset_id,
    )

    pooled_rows, dedup_log_rows = dedupe_real_rows(
        normalized_rows=normalized_rows,
        subset_order=args.subset_order,
    )

    pooled_fieldnames = [
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
        "original_run_id",
        "source_subset_id",
        "source_csv",
    ]

    dedup_fieldnames = [
        "pooled_run_id",
        "source_video_id",
        "candidate_video_id",
        "candidate_subset_id",
        "candidate_original_run_id",
        "candidate_score",
        "candidate_label",
        "candidate_score_available",
        "candidate_face_found",
        "candidate_frame_score_coverage",
        "candidate_n_frames_scored",
        "is_kept",
        "selection_reason",
        "kept_subset_id",
        "kept_original_run_id",
        "dedup_group_size",
    ]

    write_csv(args.output_csv, pooled_fieldnames, pooled_rows)
    write_csv(args.dedup_log, dedup_fieldnames, dedup_log_rows)

    before_summary = summarize_rows(normalized_rows)
    after_summary = summarize_rows(pooled_rows)

    print("Pooled FF++ calibration table built.")
    print(f"pooled_run_id: {pooled_run_id}")
    print(f"output_csv: {args.output_csv}")
    print(f"dedup_log: {args.dedup_log}")
    print("")
    print("Before real-video dedup:")
    for k, v in before_summary.items():
        print(f"  {k}: {v}")
    print("")
    print("After real-video dedup:")
    for k, v in after_summary.items():
        print(f"  {k}: {v}")
    print("")
    print(f"dedup_log_rows: {len(dedup_log_rows)}")


if __name__ == "__main__":
    main()