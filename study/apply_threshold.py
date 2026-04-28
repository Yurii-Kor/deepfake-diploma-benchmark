#!/usr/bin/env python3
import argparse
import csv
import os
from typing import Dict, List, Optional


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


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, fieldnames: List[str], rows: List[Dict]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: str, fieldnames: List[str], rows: List[Dict]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def infer_common_metadata(rows: List[Dict[str, str]]) -> Dict[str, str]:
    if not rows:
        return {}
    first = rows[0]
    keys = [
        "run_id",
        "model_id",
        "family_id",
        "checkpoint_id",
        "dataset_id",
        "subset_id",
        "split",
        "condition_code",
    ]
    return {k: first.get(k, "") for k in keys}


def _matches_input_run(reg_row: Dict[str, str], input_meta: Dict[str, str]) -> bool:
    # Strongest match: exact run_id
    run_id = input_meta.get("run_id", "")
    if run_id and reg_row.get("run_id", "") == run_id:
        return True

    # Fallback: match key provenance fields
    keys = ["model_id", "family_id", "checkpoint_id", "dataset_id", "subset_id", "split", "condition_code"]
    return all(reg_row.get(k, "") == input_meta.get(k, "") for k in keys)


def select_threshold_row(
    registry_rows: List[Dict[str, str]],
    input_meta: Dict[str, str],
    selection_rule: str,
    threshold_id: Optional[str] = None,
) -> Dict[str, str]:
    if threshold_id:
        exact = [r for r in registry_rows if r.get("threshold_id", "") == threshold_id]
        if not exact:
            raise ValueError(f"threshold_id not found in registry: {threshold_id}")
        return exact[-1]

    candidates = [r for r in registry_rows if r.get("selection_rule", "") == selection_rule]
    if not candidates:
        raise ValueError(f"No rows found in threshold registry for selection_rule={selection_rule}")

    matching = [r for r in candidates if _matches_input_run(r, input_meta)]
    if matching:
        return matching[-1]

    raise ValueError(
        "No threshold row matched the input video CSV provenance. "
        "Pass --threshold-id explicitly or verify run_id / metadata alignment."
    )


def build_master_rows(
    video_rows: List[Dict[str, str]],
    threshold_row: Dict[str, str],
) -> List[Dict]:
    threshold_value = _to_float_or_none(threshold_row.get("threshold_value"))
    if threshold_value is None:
        raise ValueError("Selected threshold row has no threshold_value")

    out_rows = []
    for r in video_rows:
        score = _to_float_or_none(r.get("score"))
        score_available = _to_bool(r.get("score_available", True))

        if score_available and score is not None:
            decision = 1 if score >= threshold_value else 0
        else:
            decision = ""

        out_rows.append({
            "run_id": r.get("run_id", ""),
            "model_id": r.get("model_id", ""),
            "family_id": r.get("family_id", ""),
            "checkpoint_id": r.get("checkpoint_id", ""),
            "dataset_id": r.get("dataset_id", ""),
            "subset_id": r.get("subset_id", ""),
            "split": r.get("split", ""),
            "condition_code": r.get("condition_code", ""),
            "video_id": r.get("video_id", ""),
            "source_video_id": r.get("source_video_id", ""),
            "score": r.get("score", ""),
            "decision": decision,
            "label": r.get("label", ""),
            "threshold_id": threshold_row.get("threshold_id", ""),
            "threshold_value": threshold_row.get("threshold_value", ""),
            "threshold_selection_rule": threshold_row.get("selection_rule", ""),
            "threshold_calibration_role": threshold_row.get("calibration_role", ""),
            "score_available": r.get("score_available", ""),
            "face_found": r.get("face_found", ""),
            "n_frames_expected": r.get("n_frames_expected", ""),
            "n_frames_scored": r.get("n_frames_scored", ""),
            "n_frames_missing": r.get("n_frames_missing", ""),
            "frame_score_coverage": r.get("frame_score_coverage", ""),
            "partial_score": r.get("partial_score", ""),
            "failure_stage": r.get("failure_stage", ""),
            "failure_reason": r.get("failure_reason", ""),
        })

    return out_rows


def summarize_master(master_rows: List[Dict]) -> Dict[str, float]:
    n_total = len(master_rows)
    n_scored = sum(1 for r in master_rows if _to_bool(r.get("score_available", True)) and str(r.get("decision", "")).strip() != "")
    n_no_score = sum(1 for r in master_rows if not _to_bool(r.get("score_available", True)))
    n_partial = sum(1 for r in master_rows if _to_bool(r.get("partial_score", False)))

    n_correct = 0
    for r in master_rows:
        d = str(r.get("decision", "")).strip()
        y = str(r.get("label", "")).strip()
        if d != "" and y != "" and d == y:
            n_correct += 1

    accuracy_scored_subset = (n_correct / n_scored) if n_scored else float("nan")

    return {
        "n_total_video_rows": n_total,
        "n_scored_video_rows": n_scored,
        "n_no_score_video_rows": n_no_score,
        "n_partial_video_rows": n_partial,
        "accuracy_on_scored_subset": accuracy_scored_subset,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, help="Video-level aggregated CSV")
    parser.add_argument("--threshold-registry-csv", required=True, help="threshold_registry.csv path")
    parser.add_argument("--output-csv", required=True, help="master_results.csv path")
    parser.add_argument(
        "--selection-rule",
        default="primary_target_fpr",
        help="Threshold selection_rule to use if --threshold-id is not provided",
    )
    parser.add_argument(
        "--threshold-id",
        default=None,
        help="Exact threshold_id to use. Overrides --selection-rule",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows to output instead of overwriting",
    )
    args = parser.parse_args()

    video_rows = read_csv_rows(args.input_csv)
    if not video_rows:
        raise ValueError("Input video CSV is empty")

    registry_rows = read_csv_rows(args.threshold_registry_csv)
    if not registry_rows:
        raise ValueError("Threshold registry CSV is empty")

    input_meta = infer_common_metadata(video_rows)
    threshold_row = select_threshold_row(
        registry_rows=registry_rows,
        input_meta=input_meta,
        selection_rule=args.selection_rule,
        threshold_id=args.threshold_id,
    )

    master_rows = build_master_rows(video_rows, threshold_row)

    fieldnames = [
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
        "decision",
        "label",
        "threshold_id",
        "threshold_value",
        "threshold_selection_rule",
        "threshold_calibration_role",
        "score_available",
        "face_found",
        "n_frames_expected",
        "n_frames_scored",
        "n_frames_missing",
        "frame_score_coverage",
        "partial_score",
        "failure_stage",
        "failure_reason",
    ]

    if args.append:
        append_csv(args.output_csv, fieldnames, master_rows)
    else:
        write_csv(args.output_csv, fieldnames, master_rows)

    summary = summarize_master(master_rows)

    print("Apply-threshold stage done.")
    print(f"Selected threshold_id: {threshold_row.get('threshold_id', '')}")
    print(f"Selected threshold_value: {threshold_row.get('threshold_value', '')}")
    print(f"Selection rule: {threshold_row.get('selection_rule', '')}")
    print(f"Calibration role: {threshold_row.get('calibration_role', '')}")
    print(f"master_results_csv: {args.output_csv}")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()