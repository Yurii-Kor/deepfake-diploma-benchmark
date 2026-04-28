#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict
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


def confusion_counts(rows: List[Dict[str, str]]) -> Tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for r in rows:
        y = _to_int_or_none(r.get("label"))
        d = _to_int_or_none(r.get("decision"))
        if y is None or d is None:
            continue
        if d == 1 and y == 1:
            tp += 1
        elif d == 1 and y == 0:
            fp += 1
        elif d == 0 and y == 0:
            tn += 1
        elif d == 0 and y == 1:
            fn += 1
    return tp, fp, tn, fn


def safe_div(num: float, den: float) -> float:
    return (num / den) if den else math.nan


def compute_threshold_metrics(scored_rows: List[Dict[str, str]]) -> Dict[str, float]:
    tp, fp, tn, fn = confusion_counts(scored_rows)

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall_tpr = safe_div(tp, tp + fn)
    fpr = safe_div(fp, fp + tn)
    tnr = safe_div(tn, tn + fp)
    fnr = safe_div(fn, fn + tp)
    f1 = safe_div(2 * precision * recall_tpr, precision + recall_tpr) if not math.isnan(precision) and not math.isnan(recall_tpr) and (precision + recall_tpr) else math.nan
    balanced_accuracy = safe_div(recall_tpr + tnr, 2) if not math.isnan(recall_tpr) and not math.isnan(tnr) else math.nan
    hter = safe_div(fpr + fnr, 2) if not math.isnan(fpr) and not math.isnan(fnr) else math.nan

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall_tpr": recall_tpr,
        "fpr": fpr,
        "tnr": tnr,
        "fnr": fnr,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "hter": hter,
    }


def compute_coverage_metrics(all_rows: List[Dict[str, str]]) -> Dict[str, float]:
    n_total_videos = len(all_rows)
    n_scored_videos = sum(1 for r in all_rows if _to_bool(r.get("score_available", True)) and str(r.get("decision", "")).strip() != "")
    n_no_score_videos = sum(1 for r in all_rows if not _to_bool(r.get("score_available", True)))
    n_face_found_videos = sum(1 for r in all_rows if _to_bool(r.get("face_found", False)))
    n_partial_videos = sum(1 for r in all_rows if _to_bool(r.get("partial_score", False)))

    score_available_rate = safe_div(n_scored_videos, n_total_videos)
    face_found_rate = safe_div(n_face_found_videos, n_total_videos)

    return {
        "n_total_videos": n_total_videos,
        "n_scored_videos": n_scored_videos,
        "n_no_score_videos": n_no_score_videos,
        "n_face_found_videos": n_face_found_videos,
        "n_partial_videos": n_partial_videos,
        "score_available_rate": score_available_rate,
        "face_found_rate": face_found_rate,
    }


def build_group_key(row: Dict[str, str]) -> Tuple[str, ...]:
    return (
        row.get("run_id", ""),
        row.get("model_id", ""),
        row.get("family_id", ""),
        row.get("checkpoint_id", ""),
        row.get("dataset_id", ""),
        row.get("subset_id", ""),
        row.get("split", ""),
        row.get("condition_code", ""),
        row.get("threshold_id", ""),
        row.get("threshold_value", ""),
        row.get("threshold_selection_rule", ""),
        row.get("threshold_calibration_role", ""),
    )


def summarize_group(rows: List[Dict[str, str]]) -> Dict:
    first = rows[0]

    scored_rows = [
        r for r in rows
        if _to_bool(r.get("score_available", True)) and _to_int_or_none(r.get("decision")) is not None
    ]

    threshold_metrics = compute_threshold_metrics(scored_rows)
    coverage_metrics = compute_coverage_metrics(rows)

    out = {
        "run_id": first.get("run_id", ""),
        "model_id": first.get("model_id", ""),
        "family_id": first.get("family_id", ""),
        "checkpoint_id": first.get("checkpoint_id", ""),
        "dataset_id": first.get("dataset_id", ""),
        "subset_id": first.get("subset_id", ""),
        "split": first.get("split", ""),
        "condition_code": first.get("condition_code", ""),
        "threshold_id": first.get("threshold_id", ""),
        "threshold_value": first.get("threshold_value", ""),
        "threshold_selection_rule": first.get("threshold_selection_rule", ""),
        "threshold_calibration_role": first.get("threshold_calibration_role", ""),
    }

    out.update(coverage_metrics)
    out.update(threshold_metrics)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, help="master_results.csv path")
    parser.add_argument("--output-csv", required=True, help="metrics_summary.csv path")
    args = parser.parse_args()

    rows = read_csv_rows(args.input_csv)
    if not rows:
        raise ValueError("Input master_results CSV is empty")

    grouped = defaultdict(list)
    for r in rows:
        grouped[build_group_key(r)].append(r)

    summary_rows = []
    for _, group_rows in grouped.items():
        summary_rows.append(summarize_group(group_rows))

    summary_rows.sort(key=lambda r: (
        r["run_id"],
        r["dataset_id"],
        r["subset_id"],
        r["condition_code"],
        r["threshold_id"],
    ))

    fieldnames = [
        "run_id",
        "model_id",
        "family_id",
        "checkpoint_id",
        "dataset_id",
        "subset_id",
        "split",
        "condition_code",
        "threshold_id",
        "threshold_value",
        "threshold_selection_rule",
        "threshold_calibration_role",
        "n_total_videos",
        "n_scored_videos",
        "n_no_score_videos",
        "n_face_found_videos",
        "n_partial_videos",
        "score_available_rate",
        "face_found_rate",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "precision",
        "recall_tpr",
        "fpr",
        "tnr",
        "fnr",
        "f1",
        "balanced_accuracy",
        "hter",
    ]

    write_csv(args.output_csv, fieldnames, summary_rows)

    print("Metrics summary stage done.")
    print(f"Input master rows: {len(rows)}")
    print(f"Output summary rows: {len(summary_rows)}")
    print(f"metrics_summary_csv: {args.output_csv}")

    for row in summary_rows:
        print("")
        print(
            f"{row['model_id']} | {row['dataset_id']} | {row['subset_id']} | "
            f"{row['condition_code']} | {row['threshold_id']}"
        )
        print(f"  n_total_videos: {row['n_total_videos']}")
        print(f"  n_scored_videos: {row['n_scored_videos']}")
        print(f"  n_no_score_videos: {row['n_no_score_videos']}")
        print(f"  n_partial_videos: {row['n_partial_videos']}")
        print(f"  score_available_rate: {row['score_available_rate']}")
        print(f"  face_found_rate: {row['face_found_rate']}")
        print(f"  accuracy: {row['accuracy']}")
        print(f"  precision: {row['precision']}")
        print(f"  recall_tpr: {row['recall_tpr']}")
        print(f"  fpr: {row['fpr']}")
        print(f"  fnr: {row['fnr']}")
        print(f"  balanced_accuracy: {row['balanced_accuracy']}")
        print(f"  hter: {row['hter']}")


if __name__ == "__main__":
    main()