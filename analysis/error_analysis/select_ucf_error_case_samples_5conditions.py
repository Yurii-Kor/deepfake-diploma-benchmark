#!/usr/bin/env python3
"""
Select representative UCF error cases for the five-condition FF++ degradation study.

This script extends the earlier C0/R1/C2 error-case sampling logic to the full
UCF threshold-transfer setup:

    C0_full_original
    R1_resize_x05
    L1_blur_sigma1
    C2_h264_crf40
    P1_platform_like

The script joins video-level master result CSV files across all five conditions,
selects representative cases for qualitative error analysis, and optionally copies
the corresponding MP4 videos into grouped folders.

Important sampling rules:
    - Real-video groups are selected once over unique original videos, because the
      same real videos appear in each FF++ manipulation-family subset.
    - Fake-video groups are selected per manipulation subset, because fake videos
      are family-specific.
    - Groups are processed in priority order A -> H.
    - Once a video/case is selected in an earlier group, it is not selected again
      in later groups. This keeps the final qualitative sample non-overlapping.

Final qualitative groups:
    A_persistent_FP_all
    B_L1_induced_FP
    C_P1_induced_FP
    D_C2_only_FN
    E_P1_only_FN
    F_C2_P1_FN
    G_near_threshold_real
    H_near_threshold_fake

Default mode writes CSV/metadata only.
Use --copy-videos only when the clean/degraded video folders are available.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple


THRESHOLD_DEFAULT = 0.3905336612369865

CONDITIONS: List[Tuple[str, str]] = [
    ("C0_full_original", "C0"),
    ("R1_resize_x05", "R1_resize_x05"),
    ("L1_blur_sigma1", "L1_blur_sigma1"),
    ("C2_h264_crf40", "C2_h264_crf40"),
    ("P1_platform_like", "P1_platform_like"),
]

SUBSETS = ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]

SUBSET_TO_VIDEO_FOLDER = {
    "FF-DF": "Deepfakes",
    "FF-F2F": "Face2Face",
    "FF-FS": "FaceSwap",
    "FF-NT": "NeuralTextures",
}

REAL_ONLY_GROUPS = {
    "A_persistent_FP_all",
    "B_L1_induced_FP",
    "C_P1_induced_FP",
    "G_near_threshold_real",
}

DEFAULT_MASTER_ROOT = Path("/home/sceuser/deepfake_lab/study_outputs/master")
DEFAULT_OUTPUT_ROOT = Path("/home/sceuser/deepfake_lab/error_case_samples_ucf_5conditions")
DEFAULT_CLEAN_DATA_ROOT = Path("/home/sceuser/deepfake_lab/deepfake_data/FaceForensics++")
DEFAULT_DEGRADED_BASE = Path("/home/sceuser/deepfake_lab/deepfake_data_degraded")

DEFAULT_MAX_PER_GROUP_PER_SUBSET = 3
DEFAULT_NEAR_THRESHOLD_MARGIN = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select UCF error-case samples across C0/R1/L1/C2/P1 conditions."
    )

    parser.add_argument(
        "--master-root",
        type=Path,
        default=DEFAULT_MASTER_ROOT,
        help="Directory with ucf_<condition>_<subset>_test_master_results.csv files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory for joined tables, selected cases, and optional copied videos.",
    )
    parser.add_argument(
        "--clean-data-root",
        type=Path,
        default=DEFAULT_CLEAN_DATA_ROOT,
        help="Clean FaceForensics++ root used for C0_full_original videos.",
    )
    parser.add_argument(
        "--degraded-base",
        type=Path,
        default=DEFAULT_DEGRADED_BASE,
        help="Base directory containing degraded condition folders.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD_DEFAULT,
        help="Frozen clean C0 threshold used by the UCF threshold-transfer experiment.",
    )
    parser.add_argument(
        "--near-threshold-margin",
        type=float,
        default=DEFAULT_NEAR_THRESHOLD_MARGIN,
        help="Absolute score distance from threshold for near-threshold cases.",
    )
    parser.add_argument(
        "--max-per-group-per-subset",
        type=int,
        default=DEFAULT_MAX_PER_GROUP_PER_SUBSET,
        help="Maximum selected cases per group and FF++ subset.",
    )
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy matching MP4 files into grouped case directories.",
    )

    return parser.parse_args()


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = float(text)
    except ValueError:
        return None

    if math.isnan(parsed):
        return None

    return parsed


def to_int(value: object) -> Optional[int]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return int(float(text))
    except ValueError:
        return None


def normalize_video_id(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"\.mp4$", "", text)
    text = re.sub(r"\.avi$", "", text)
    return text


def safe_name(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def first_existing_column(
    fieldnames: Iterable[str],
    candidates: Iterable[str],
) -> Optional[str]:
    fieldname_set = set(fieldnames)

    for candidate in candidates:
        if candidate in fieldname_set:
            return candidate

    return None


def infer_required_columns(path: Path, fieldnames: List[str]) -> Dict[str, str]:
    video_col = first_existing_column(
        fieldnames,
        ["video_id", "video", "video_name", "filename", "file_name"],
    )
    label_col = first_existing_column(
        fieldnames,
        ["label", "true_label", "target", "y_true"],
    )
    score_col = first_existing_column(
        fieldnames,
        ["score", "video_score", "avg_score", "mean_score", "fake_score", "probability"],
    )

    missing = []
    if video_col is None:
        missing.append("video_id")
    if label_col is None:
        missing.append("label")
    if score_col is None:
        missing.append("score")

    if missing:
        raise ValueError(
            f"Missing required columns {missing} in {path}. "
            f"Available columns: {fieldnames}"
        )

    return {
        "video": video_col,
        "label": label_col,
        "score": score_col,
    }


def read_master_file(
    master_root: Path,
    condition: str,
    subset: str,
    threshold: float,
) -> Dict[Tuple[str, str, str], dict]:
    path = master_root / f"ucf_{condition}_{subset}_test_master_results.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing master file: {path}")

    rows: Dict[Tuple[str, str, str], dict] = {}

    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []

        required = infer_required_columns(path, fieldnames)

        subset_col = first_existing_column(fieldnames, ["subset_id", "subset"])
        source_col = first_existing_column(fieldnames, ["source_video_id", "source_id", "source_video"])
        decision_col = first_existing_column(fieldnames, ["decision", "pred_label", "prediction", "pred", "y_pred"])
        threshold_col = first_existing_column(fieldnames, ["threshold_value", "threshold"])
        frames_col = first_existing_column(fieldnames, ["n_frames_scored", "num_frames_scored", "frames_scored"])
        missing_frames_col = first_existing_column(fieldnames, ["n_frames_missing", "num_frames_missing"])
        partial_col = first_existing_column(fieldnames, ["partial_score", "is_partial", "partial"])

        for raw in reader:
            subset_id = raw[subset_col] if subset_col else subset
            video_id = normalize_video_id(raw[required["video"]])
            source_video_id = normalize_video_id(raw[source_col]) if source_col else video_id

            label = to_int(raw[required["label"]])
            score = to_float(raw[required["score"]])

            if label is None or score is None:
                continue

            decision = to_int(raw[decision_col]) if decision_col else None
            if decision is None:
                decision = 1 if score >= threshold else 0

            threshold_value = to_float(raw[threshold_col]) if threshold_col else None
            if threshold_value is None:
                threshold_value = threshold

            key = (subset_id, video_id, str(label))

            rows[key] = {
                "subset_id": subset_id,
                "video_id": video_id,
                "source_video_id": source_video_id,
                "label": label,
                "score": score,
                "decision": decision,
                "threshold_value": threshold_value,
                "n_frames_scored": raw[frames_col] if frames_col else "",
                "n_frames_missing": raw[missing_frames_col] if missing_frames_col else "",
                "partial_score": raw[partial_col] if partial_col else "",
            }

    return rows


def build_joined_rows(master_root: Path, threshold: float) -> List[dict]:
    joined_rows: List[dict] = []

    for subset in SUBSETS:
        condition_rows = {
            condition: read_master_file(master_root, condition, subset, threshold)
            for condition, _ in CONDITIONS
        }

        common_keys = None

        for rows in condition_rows.values():
            if common_keys is None:
                common_keys = set(rows.keys())
            else:
                common_keys &= set(rows.keys())

        if not common_keys:
            continue

        for key in sorted(common_keys):
            subset_id, video_id, label_text = key
            label = int(label_text)

            base_condition = CONDITIONS[0][0]
            base_row = condition_rows[base_condition][key]

            joined = {
                "subset_id": subset_id,
                "video_id": video_id,
                "source_video_id": base_row["source_video_id"],
                "label": label,
                "threshold_value": base_row["threshold_value"],
            }

            for condition, short_name in CONDITIONS:
                current = condition_rows[condition][key]

                joined[f"score_{short_name}"] = current["score"]
                joined[f"decision_{short_name}"] = current["decision"]
                joined[f"n_frames_scored_{short_name}"] = current["n_frames_scored"]
                joined[f"n_frames_missing_{short_name}"] = current["n_frames_missing"]
                joined[f"partial_score_{short_name}"] = current["partial_score"]

            add_derived_columns(joined)
            joined_rows.append(joined)

    return joined_rows


def row_score(row: dict, short_name: str) -> float:
    value = to_float(row.get(f"score_{short_name}"))

    if value is None:
        raise ValueError(f"Missing score_{short_name} in row: {row}")

    return value


def row_decision(row: dict, short_name: str) -> int:
    value = to_int(row.get(f"decision_{short_name}"))

    if value is None:
        raise ValueError(f"Missing decision_{short_name} in row: {row}")

    return value


def row_threshold(row: dict) -> float:
    value = to_float(row.get("threshold_value"))

    if value is None:
        return THRESHOLD_DEFAULT

    return value


def near_threshold_distance(row: dict) -> float:
    threshold_value = row_threshold(row)

    return min(
        abs(row_score(row, short_name) - threshold_value)
        for _, short_name in CONDITIONS
    )


def add_derived_columns(row: dict) -> None:
    row["near_threshold_distance"] = near_threshold_distance(row)

    row["score_drop_C0_to_C2"] = (
        row_score(row, "C0") - row_score(row, "C2_h264_crf40")
    )
    row["score_drop_C0_to_P1"] = (
        row_score(row, "C0") - row_score(row, "P1_platform_like")
    )
    row["score_jump_C0_to_L1"] = (
        row_score(row, "L1_blur_sigma1") - row_score(row, "C0")
    )
    row["score_jump_C0_to_P1"] = (
        row_score(row, "P1_platform_like") - row_score(row, "C0")
    )
    row["score_drop_C0_to_mean_C2_P1"] = row_score(row, "C0") - (
        row_score(row, "C2_h264_crf40") + row_score(row, "P1_platform_like")
    ) / 2.0


def group_definitions(
    near_threshold_margin: float,
) -> Dict[str, Tuple[Callable[[dict], bool], str, bool, str]]:
    def is_real(row: dict) -> bool:
        return int(row["label"]) == 0

    def is_fake(row: dict) -> bool:
        return int(row["label"]) == 1

    return {
        "A_persistent_FP_all": (
            lambda row: is_real(row)
            and row_decision(row, "C0") == 1
            and row_decision(row, "R1_resize_x05") == 1
            and row_decision(row, "L1_blur_sigma1") == 1
            and row_decision(row, "C2_h264_crf40") == 1
            and row_decision(row, "P1_platform_like") == 1,
            "score_jump_C0_to_P1",
            False,
            "Real video: label=0, but decision=1 under all five conditions.",
        ),
        "B_L1_induced_FP": (
            lambda row: is_real(row)
            and row_decision(row, "C0") == 0
            and row_decision(row, "R1_resize_x05") == 0
            and row_decision(row, "L1_blur_sigma1") == 1,
            "score_jump_C0_to_L1",
            False,
            "Real video: C0/R1 are correct real, but L1 becomes false positive.",
        ),
        "C_P1_induced_FP": (
            lambda row: is_real(row)
            and row_decision(row, "C0") == 0
            and row_decision(row, "R1_resize_x05") == 0
            and row_decision(row, "P1_platform_like") == 1,
            "score_jump_C0_to_P1",
            False,
            "Real video: C0/R1 are correct real, but P1 becomes false positive.",
        ),
        "D_C2_only_FN": (
            lambda row: is_fake(row)
            and row_decision(row, "C0") == 1
            and row_decision(row, "R1_resize_x05") == 1
            and row_decision(row, "L1_blur_sigma1") == 1
            and row_decision(row, "C2_h264_crf40") == 0
            and row_decision(row, "P1_platform_like") == 1,
            "score_drop_C0_to_C2",
            False,
            "Fake video: C0/R1/L1/P1 are correct fake, but C2 becomes false negative.",
        ),
        "E_P1_only_FN": (
            lambda row: is_fake(row)
            and row_decision(row, "C0") == 1
            and row_decision(row, "R1_resize_x05") == 1
            and row_decision(row, "L1_blur_sigma1") == 1
            and row_decision(row, "C2_h264_crf40") == 1
            and row_decision(row, "P1_platform_like") == 0,
            "score_drop_C0_to_P1",
            False,
            "Fake video: C0/R1/L1/C2 are correct fake, but P1 becomes false negative.",
        ),
        "F_C2_P1_FN": (
            lambda row: is_fake(row)
            and row_decision(row, "C0") == 1
            and row_decision(row, "R1_resize_x05") == 1
            and row_decision(row, "L1_blur_sigma1") == 1
            and row_decision(row, "C2_h264_crf40") == 0
            and row_decision(row, "P1_platform_like") == 0,
            "score_drop_C0_to_mean_C2_P1",
            False,
            "Fake video: C0/R1/L1 are correct fake, but both C2 and P1 become false negative.",
        ),
        "G_near_threshold_real": (
            lambda row: is_real(row)
            and near_threshold_distance(row) <= near_threshold_margin,
            "near_threshold_distance",
            True,
            "Real video score is close to the fixed threshold in at least one condition.",
        ),
        "H_near_threshold_fake": (
            lambda row: is_fake(row)
            and near_threshold_distance(row) <= near_threshold_margin,
            "near_threshold_distance",
            True,
            "Fake video score is close to the fixed threshold in at least one condition.",
        ),
    }


def case_identity(row: dict) -> tuple:
    if int(row["label"]) == 0:
        return (
            "real",
            "FF-real",
            row.get("source_video_id") or row["video_id"],
        )

    return (
        "fake",
        row["subset_id"],
        row["video_id"],
    )


def select_unique_cases(
    rows: List[dict],
    metric_name: str,
    ascending: bool,
    limit: int,
    unique_key: Callable[[dict], tuple],
) -> List[dict]:
    ranked = sorted(
        rows,
        key=lambda row: (row.get(metric_name, float("inf")), row["video_id"]),
        reverse=not ascending,
    )

    selected: List[dict] = []
    seen = set()

    for row in ranked:
        key = unique_key(row)

        if key in seen:
            continue

        seen.add(key)
        selected.append(row)

        if len(selected) >= limit:
            break

    return selected


def select_cases(
    joined_rows: List[dict],
    near_threshold_margin: float,
    max_per_group_per_subset: int,
) -> Tuple[List[dict], List[dict]]:
    definitions = group_definitions(near_threshold_margin)

    selected_rows: List[dict] = []
    summary_rows: List[dict] = []
    selected_identities = set()
    case_number = 1

    for group_name, (predicate, metric_name, ascending, logic) in definitions.items():
        candidates = [
            row for row in joined_rows
            if predicate(row) and case_identity(row) not in selected_identities
        ]

        if group_name in REAL_ONLY_GROUPS:
            unique_candidate_count = len({case_identity(row) for row in candidates})

            selected_group = select_unique_cases(
                rows=candidates,
                metric_name=metric_name,
                ascending=ascending,
                limit=max_per_group_per_subset,
                unique_key=case_identity,
            )

            summary_rows.append(
                {
                    "case_group": group_name,
                    "subset_id": "FF-real",
                    "candidate_count": unique_candidate_count,
                    "selected_count": len(selected_group),
                    "selection_metric": metric_name,
                    "ascending": ascending,
                    "logic": logic,
                }
            )

            for row in selected_group:
                identity = case_identity(row)
                selected_identities.add(identity)

                selected = dict(row)
                selected["case_id"] = f"{case_number:03d}"
                selected["case_group"] = group_name
                selected["selection_scope"] = "FF-real"
                selected["selection_metric"] = metric_name
                selected["selection_metric_value"] = selected.get(metric_name, "")
                selected_rows.append(selected)
                case_number += 1

            continue

        for subset in SUBSETS:
            subset_candidates = [
                row for row in candidates
                if row["subset_id"] == subset
                and case_identity(row) not in selected_identities
            ]

            selected_subset = select_unique_cases(
                rows=subset_candidates,
                metric_name=metric_name,
                ascending=ascending,
                limit=max_per_group_per_subset,
                unique_key=case_identity,
            )

            summary_rows.append(
                {
                    "case_group": group_name,
                    "subset_id": subset,
                    "candidate_count": len(subset_candidates),
                    "selected_count": len(selected_subset),
                    "selection_metric": metric_name,
                    "ascending": ascending,
                    "logic": logic,
                }
            )

            for row in selected_subset:
                identity = case_identity(row)
                selected_identities.add(identity)

                selected = dict(row)
                selected["case_id"] = f"{case_number:03d}"
                selected["case_group"] = group_name
                selected["selection_scope"] = subset
                selected["selection_metric"] = metric_name
                selected["selection_metric_value"] = selected.get(metric_name, "")
                selected_rows.append(selected)
                case_number += 1

    return selected_rows, summary_rows


def video_folder_for_case(subset: str, label: int) -> str:
    if label == 0:
        return "original"

    if subset not in SUBSET_TO_VIDEO_FOLDER:
        raise ValueError(f"Unknown subset: {subset}")

    return SUBSET_TO_VIDEO_FOLDER[subset]


def resolve_video_path(
    condition: str,
    subset: str,
    label: int,
    video_id: str,
    source_video_id: str,
    clean_data_root: Path,
    degraded_base: Path,
) -> Path:
    folder = video_folder_for_case(subset, label)

    if condition == "C0_full_original":
        root = clean_data_root
    else:
        root = degraded_base / condition / "FaceForensics++"

    video_ids = [
        normalize_video_id(video_id),
        normalize_video_id(source_video_id),
    ]

    for current_video_id in dict.fromkeys(video_ids):
        candidate = root / folder / f"{current_video_id}.mp4"

        if candidate.exists():
            return candidate

    return root / folder / f"{video_ids[0]}.mp4"


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_case_videos(
    row: dict,
    case_dir: Path,
    clean_data_root: Path,
    degraded_base: Path,
    copy_videos: bool,
) -> dict:
    copy_info: dict = {}

    subset = row["subset_id"]
    label = int(row["label"])
    video_id = row["video_id"]
    source_video_id = row.get("source_video_id", video_id)

    for condition, short_name in CONDITIONS:
        source = resolve_video_path(
            condition=condition,
            subset=subset,
            label=label,
            video_id=video_id,
            source_video_id=source_video_id,
            clean_data_root=clean_data_root,
            degraded_base=degraded_base,
        )

        destination_name = f"{short_name}_{subset}_{safe_name(video_id)}.mp4"
        destination = case_dir / destination_name

        copy_info[f"{short_name}_video_src"] = str(source)

        if not source.exists():
            copy_info[f"{short_name}_video_copied"] = "missing"
            copy_info[f"{short_name}_video_dst"] = ""
            continue

        if copy_videos:
            shutil.copy2(source, destination)
            copy_info[f"{short_name}_video_copied"] = "yes"
            copy_info[f"{short_name}_video_dst"] = str(destination)
        else:
            copy_info[f"{short_name}_video_copied"] = "not_requested"
            copy_info[f"{short_name}_video_dst"] = ""

    return copy_info


def materialize_selected_cases(
    selected_rows: List[dict],
    output_root: Path,
    clean_data_root: Path,
    degraded_base: Path,
    copy_videos: bool,
) -> List[dict]:
    copied_rows: List[dict] = []

    for row in selected_rows:
        case_id = row["case_id"]
        group = row["case_group"]
        scope = row.get("selection_scope") or row["subset_id"]
        video_id = safe_name(row["video_id"])

        case_dir = output_root / group / safe_name(scope) / f"case_{case_id}_{video_id}"
        case_dir.mkdir(parents=True, exist_ok=True)

        copied_row = dict(row)

        copy_info = copy_case_videos(
            row=row,
            case_dir=case_dir,
            clean_data_root=clean_data_root,
            degraded_base=degraded_base,
            copy_videos=copy_videos,
        )

        copied_row.update(copy_info)
        copied_row["case_dir"] = str(case_dir)

        write_csv(case_dir / "metadata.csv", [copied_row])
        copied_rows.append(copied_row)

    return copied_rows


def print_group_summary(summary_rows: List[dict]) -> None:
    print("Group summary:")

    for row in summary_rows:
        candidate_count = int(row["candidate_count"])
        selected_count = int(row["selected_count"])

        if candidate_count == 0 and selected_count == 0:
            continue

        print(
            f'{row["case_group"]} | {row["subset_id"]}: '
            f"candidates={candidate_count}, selected={selected_count}"
        )


def main() -> None:
    args = parse_args()

    master_root = args.master_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    clean_data_root = args.clean_data_root.expanduser()
    degraded_base = args.degraded_base.expanduser()

    output_root.mkdir(parents=True, exist_ok=True)

    joined_rows = build_joined_rows(
        master_root=master_root,
        threshold=args.threshold,
    )

    selected_rows, summary_rows = select_cases(
        joined_rows=joined_rows,
        near_threshold_margin=args.near_threshold_margin,
        max_per_group_per_subset=args.max_per_group_per_subset,
    )

    copied_rows = materialize_selected_cases(
        selected_rows=selected_rows,
        output_root=output_root,
        clean_data_root=clean_data_root,
        degraded_base=degraded_base,
        copy_videos=args.copy_videos,
    )

    joined_csv = output_root / "all_joined_UCF_C0_R1_L1_C2_P1_cases.csv"
    selected_csv = output_root / "selected_ucf_error_cases_5conditions.csv"
    summary_csv = output_root / "group_summary_ucf_5conditions.csv"
    copied_csv = output_root / "copied_ucf_error_cases_5conditions.csv"

    write_csv(joined_csv, joined_rows)
    write_csv(selected_csv, selected_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(copied_csv, copied_rows)

    print("UCF five-condition error-case sampling finished.")
    print(f"Master root: {master_root}")
    print(f"Output root: {output_root}")
    print(f"Copy videos: {args.copy_videos}")
    print(f"Joined rows: {len(joined_rows)}")
    print(f"Selected rows: {len(selected_rows)}")
    print()
    print("Output files:")
    print(f"- {joined_csv}")
    print(f"- {selected_csv}")
    print(f"- {summary_csv}")
    print(f"- {copied_csv}")
    print()

    print_group_summary(summary_rows)


if __name__ == "__main__":
    main()