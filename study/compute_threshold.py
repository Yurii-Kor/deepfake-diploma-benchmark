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


def append_csv(path: str, fieldnames: List[str], rows: List[Dict]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def filter_scored_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    scored = []
    for r in rows:
        score = _to_float_or_none(r.get("score"))
        score_available = _to_bool(r.get("score_available", True))
        if score_available and score is not None:
            scored.append(r)
    return scored


def extract_scores_and_labels(rows: List[Dict[str, str]]) -> Tuple[List[float], List[int]]:
    scores, labels = [], []
    for r in rows:
        score = _to_float_or_none(r.get("score"))
        label = _to_int_or_none(r.get("label"))
        if score is None or label is None:
            continue
        scores.append(score)
        labels.append(label)
    return scores, labels


def empirical_fpr(real_scores: List[float], threshold: float) -> float:
    if not real_scores:
        return math.nan
    fp = sum(1 for s in real_scores if s >= threshold)
    return fp / len(real_scores)


def empirical_fnr(fake_scores: List[float], threshold: float) -> float:
    if not fake_scores:
        return math.nan
    fn = sum(1 for s in fake_scores if s < threshold)
    return fn / len(fake_scores)


def accuracy(scores: List[float], labels: List[int], threshold: float) -> float:
    if not scores:
        return math.nan
    correct = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= threshold else 0
        if pred == y:
            correct += 1
    return correct / len(scores)


def precision(scores: List[float], labels: List[int], threshold: float) -> float:
    tp = 0
    fp = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= threshold else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
    denom = tp + fp
    return (tp / denom) if denom else 0.0


def recall(scores: List[float], labels: List[int], threshold: float) -> float:
    tp = 0
    fn = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= threshold else 0
        if y == 1 and pred == 1:
            tp += 1
        elif y == 1 and pred == 0:
            fn += 1
    denom = tp + fn
    return (tp / denom) if denom else 0.0


def f1_score(scores: List[float], labels: List[int], threshold: float) -> float:
    p = precision(scores, labels, threshold)
    r = recall(scores, labels, threshold)
    denom = p + r
    return (2 * p * r / denom) if denom else 0.0


def unique_threshold_candidates(scores: List[float]) -> List[float]:
    if not scores:
        return []
    uniq = sorted(set(scores))
    mx = uniq[-1]
    # add one threshold slightly above max so FPR can become exactly 0
    eps = max(1e-12, abs(mx) * 1e-12)
    uniq.append(mx + eps)
    return uniq


def select_primary_low_fpr_threshold(real_scores: List[float], target_fpr: float) -> Tuple[float, float]:
    if not real_scores:
        raise ValueError("No real scored videos available for primary threshold selection.")

    candidates = unique_threshold_candidates(real_scores)
    for t in candidates:
        fpr = empirical_fpr(real_scores, t)
        if fpr <= target_fpr:
            return t, fpr

    # Should never happen because max+eps yields FPR 0, but keep safe fallback.
    t = max(real_scores) + 1e-12
    return t, empirical_fpr(real_scores, t)


def select_eer_like_threshold(real_scores: List[float], fake_scores: List[float]) -> Tuple[float, float, float]:
    if not real_scores or not fake_scores:
        raise ValueError("Need both real and fake scored videos for EER-like threshold.")

    candidates = unique_threshold_candidates(real_scores + fake_scores)
    best_t = None
    best_gap = None
    best_fpr = None
    best_fnr = None

    for t in candidates:
        fpr = empirical_fpr(real_scores, t)
        fnr = empirical_fnr(fake_scores, t)
        gap = abs(fpr - fnr)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_t = t
            best_fpr = fpr
            best_fnr = fnr

    return best_t, best_fpr, best_fnr


def select_best_f1_threshold(scores: List[float], labels: List[int]) -> Tuple[float, float]:
    if not scores:
        raise ValueError("No scored videos available for Best-F1 threshold.")

    candidates = unique_threshold_candidates(scores)
    best_t = None
    best_f1 = None

    for t in candidates:
        f1 = f1_score(scores, labels, t)
        if best_f1 is None or f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


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


def build_threshold_registry_rows(
    metadata: Dict[str, str],
    input_csv: str,
    calibration_role: str,
    target_fpr: float,
    primary_threshold: float,
    primary_realized_fpr: float,
    eer_threshold: float,
    eer_fpr: float,
    eer_fnr: float,
    best_f1_threshold: float,
    best_f1_value: float,
    n_total_scored: int,
    n_real_scored: int,
    n_fake_scored: int,
) -> List[Dict]:
    ts = datetime.now().isoformat(timespec="seconds")
    base = {
        "created_at": ts,
        "run_id": metadata.get("run_id", ""),
        "model_id": metadata.get("model_id", ""),
        "family_id": metadata.get("family_id", ""),
        "checkpoint_id": metadata.get("checkpoint_id", ""),
        "dataset_id": metadata.get("dataset_id", ""),
        "subset_id": metadata.get("subset_id", ""),
        "split": metadata.get("split", ""),
        "condition_code": metadata.get("condition_code", ""),
        "calibration_role": calibration_role,
        "score_direction": "higher_means_more_fake",
        "source_csv": input_csv,
        "n_total_scored": n_total_scored,
        "n_real_scored": n_real_scored,
        "n_fake_scored": n_fake_scored,
    }

    return [
        {
            **base,
            "threshold_id": f"{metadata.get('model_id','model')}_primary_fpr005_{metadata.get('run_id','run')}",
            "selection_rule": "primary_target_fpr",
            "target_fpr": target_fpr,
            "threshold_value": primary_threshold,
            "realized_fpr_on_real": primary_realized_fpr,
            "reference_value": "",
            "notes": "Primary low-FPR operating point; smoke-only if not calibrated on final clean FF++ validation universe.",
        },
        {
            **base,
            "threshold_id": f"{metadata.get('model_id','model')}_eer_like_{metadata.get('run_id','run')}",
            "selection_rule": "eer_like_reference",
            "target_fpr": "",
            "threshold_value": eer_threshold,
            "realized_fpr_on_real": eer_fpr,
            "reference_value": eer_fnr,
            "notes": "Supplementary balanced reference point.",
        },
        {
            **base,
            "threshold_id": f"{metadata.get('model_id','model')}_best_f1_{metadata.get('run_id','run')}",
            "selection_rule": "best_f1_reference",
            "target_fpr": "",
            "threshold_value": best_f1_threshold,
            "realized_fpr_on_real": empirical_fpr_for_placeholder(metadata),  # replaced below
            "reference_value": best_f1_value,
            "notes": "Supplementary descriptive reference only.",
        },
    ]


def empirical_fpr_for_placeholder(metadata: Dict[str, str]):
    # Will be replaced after row construction; kept only to keep row shape stable.
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, help="Video-level scored CSV")
    parser.add_argument("--output-csv", required=True, help="threshold_registry.csv path")
    parser.add_argument("--target-fpr", type=float, default=0.05, help="Primary target FPR on real videos")
    parser.add_argument(
        "--calibration-role",
        default="smoke_check",
        choices=["smoke_check", "clean_validation_final"],
        help="Use smoke_check for debugging, clean_validation_final for final calibration runs",
    )
    args = parser.parse_args()

    rows = read_csv_rows(args.input_csv)
    scored_rows = filter_scored_rows(rows)
    if not scored_rows:
        raise ValueError("No scored rows found in input CSV.")

    metadata = infer_common_metadata(scored_rows)
    scores, labels = extract_scores_and_labels(scored_rows)

    real_scores = [s for s, y in zip(scores, labels) if y == 0]
    fake_scores = [s for s, y in zip(scores, labels) if y == 1]

    if not real_scores:
        raise ValueError("No real scored videos found. Cannot compute primary low-FPR threshold.")
    if not fake_scores:
        raise ValueError("No fake scored videos found. Cannot compute reference thresholds robustly.")

    primary_threshold, primary_realized_fpr = select_primary_low_fpr_threshold(
        real_scores=real_scores,
        target_fpr=args.target_fpr,
    )

    eer_threshold, eer_fpr, eer_fnr = select_eer_like_threshold(
        real_scores=real_scores,
        fake_scores=fake_scores,
    )

    best_f1_threshold, best_f1_value = select_best_f1_threshold(
        scores=scores,
        labels=labels,
    )

    registry_rows = build_threshold_registry_rows(
        metadata=metadata,
        input_csv=args.input_csv,
        calibration_role=args.calibration_role,
        target_fpr=args.target_fpr,
        primary_threshold=primary_threshold,
        primary_realized_fpr=primary_realized_fpr,
        eer_threshold=eer_threshold,
        eer_fpr=eer_fpr,
        eer_fnr=eer_fnr,
        best_f1_threshold=best_f1_threshold,
        best_f1_value=best_f1_value,
        n_total_scored=len(scored_rows),
        n_real_scored=len(real_scores),
        n_fake_scored=len(fake_scores),
    )

    # Fill realized FPR for Best-F1 row after registry row creation.
    for row in registry_rows:
        if row["selection_rule"] == "best_f1_reference":
            row["realized_fpr_on_real"] = empirical_fpr(real_scores, row["threshold_value"])

    fieldnames = [
        "created_at",
        "threshold_id",
        "run_id",
        "model_id",
        "family_id",
        "checkpoint_id",
        "dataset_id",
        "subset_id",
        "split",
        "condition_code",
        "calibration_role",
        "selection_rule",
        "target_fpr",
        "threshold_value",
        "realized_fpr_on_real",
        "reference_value",
        "score_direction",
        "source_csv",
        "n_total_scored",
        "n_real_scored",
        "n_fake_scored",
        "notes",
    ]

    append_csv(args.output_csv, fieldnames, registry_rows)

    print("Threshold computation done.")
    print(f"Input video rows (scored subset): {len(scored_rows)}")
    print(f"Real scored videos: {len(real_scores)}")
    print(f"Fake scored videos: {len(fake_scores)}")
    print("")
    print("Primary threshold")
    print(f"  rule: target FPR = {args.target_fpr}")
    print(f"  threshold: {primary_threshold}")
    print(f"  realized FPR on real videos: {primary_realized_fpr}")
    print("")
    print("Reference thresholds")
    print(f"  eer_like_threshold: {eer_threshold} (FPR={eer_fpr}, FNR={eer_fnr})")
    print(f"  best_f1_threshold: {best_f1_threshold} (F1={best_f1_value})")
    print("")
    print(f"threshold_registry_csv: {args.output_csv}")


if __name__ == "__main__":
    main()