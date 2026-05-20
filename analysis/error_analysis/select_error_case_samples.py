#!/usr/bin/env python3
import csv
import shutil
from pathlib import Path


OUT_ROOT = Path("/home/sceuser/deepfake_lab/study_outputs")
MASTER_ROOT = OUT_ROOT / "master"

CLEAN_DATA_ROOT = Path("/home/sceuser/deepfake_lab/deepfake_data/FaceForensics++")
DEGRADED_BASE = Path("/home/sceuser/deepfake_lab/deepfake_data_degraded")

DEST_ROOT = Path("/home/sceuser/deepfake_lab/error_case_samples")

CONDITIONS = ["C0", "R1_resize_x05", "C2_h264_crf40"]
SUBSETS = ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]

SUBSET_TO_FOLDER = {
    "FF-DF": "Deepfakes",
    "FF-F2F": "Face2Face",
    "FF-FS": "FaceSwap",
    "FF-NT": "NeuralTextures",
}

MAX_PER_GROUP_PER_SUBSET = 3
NEAR_THRESHOLD_MARGIN = 0.03


def read_master(condition, subset):
    path = MASTER_ROOT / f"ucf_{condition}_{subset}_test_master_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing master file: {path}")

    rows = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["subset_id"], row["video_id"], row["label"])
            rows[key] = {
                "condition": condition,
                "subset_id": row["subset_id"],
                "video_id": row["video_id"],
                "source_video_id": row.get("source_video_id", ""),
                "label": int(row["label"]),
                "score": float(row["score"]) if row["score"] else None,
                "decision": int(row["decision"]) if row["decision"] else None,
                "threshold_value": float(row["threshold_value"]) if row["threshold_value"] else None,
                "score_available": row.get("score_available", ""),
                "face_found": row.get("face_found", ""),
                "n_frames_scored": row.get("n_frames_scored", ""),
                "n_frames_missing": row.get("n_frames_missing", ""),
                "partial_score": row.get("partial_score", ""),
            }
    return rows


def resolve_video_path(condition, subset, label, video_id):
    folder = "original" if label == 0 else SUBSET_TO_FOLDER[subset]

    if condition == "C0":
        root = CLEAN_DATA_ROOT
    else:
        root = DEGRADED_BASE / condition / "FaceForensics++"

    candidates = [
        root / folder / f"{video_id}.mp4",
        root / folder / video_id,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def copy_case_videos(case_dir, subset, label, video_id):
    copied = {}

    for condition in CONDITIONS:
        src = resolve_video_path(condition, subset, label, video_id)
        dst_name = f"{condition}_{subset}_{video_id}.mp4"
        dst = case_dir / dst_name

        if src.exists():
            shutil.copy2(src, dst)
            copied[f"{condition}_video_copied"] = "yes"
            copied[f"{condition}_video_src"] = str(src)
            copied[f"{condition}_video_dst"] = str(dst)
        else:
            copied[f"{condition}_video_copied"] = "no"
            copied[f"{condition}_video_src"] = str(src)
            copied[f"{condition}_video_dst"] = ""

    return copied


def write_case_metadata(case_dir, case_row):
    path = case_dir / "metadata.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_row.keys()))
        writer.writeheader()
        writer.writerow(case_row)


def build_joined_rows():
    joined = []

    for subset in SUBSETS:
        per_condition = {
            condition: read_master(condition, subset)
            for condition in CONDITIONS
        }

        common_keys = set(per_condition["C0"].keys())
        common_keys &= set(per_condition["R1_resize_x05"].keys())
        common_keys &= set(per_condition["C2_h264_crf40"].keys())

        for key in sorted(common_keys):
            subset_id, video_id, label_str = key
            label = int(label_str)

            c0 = per_condition["C0"][key]
            r1 = per_condition["R1_resize_x05"][key]
            c2 = per_condition["C2_h264_crf40"][key]

            row = {
                "subset_id": subset_id,
                "video_id": video_id,
                "source_video_id": c0["source_video_id"],
                "label": label,

                "score_C0": c0["score"],
                "decision_C0": c0["decision"],

                "score_R1_resize_x05": r1["score"],
                "decision_R1_resize_x05": r1["decision"],

                "score_C2_h264_crf40": c2["score"],
                "decision_C2_h264_crf40": c2["decision"],

                "threshold_value": c0["threshold_value"],

                "n_frames_scored_C0": c0["n_frames_scored"],
                "n_frames_scored_R1_resize_x05": r1["n_frames_scored"],
                "n_frames_scored_C2_h264_crf40": c2["n_frames_scored"],

                "partial_score_C0": c0["partial_score"],
                "partial_score_R1_resize_x05": r1["partial_score"],
                "partial_score_C2_h264_crf40": c2["partial_score"],
            }

            joined.append(row)

    return joined


def score_drop(row):
    return row["score_C0"] - row["score_C2_h264_crf40"]


def near_threshold_distance(row):
    threshold = row["threshold_value"]
    return min(
        abs(row["score_C0"] - threshold),
        abs(row["score_R1_resize_x05"] - threshold),
        abs(row["score_C2_h264_crf40"] - threshold),
    )


def assign_groups(row):
    label = row["label"]

    d0 = row["decision_C0"]
    d1 = row["decision_R1_resize_x05"]
    d2 = row["decision_C2_h264_crf40"]

    groups = []

    if label == 0 and d0 == 1 and d1 == 1 and d2 == 1:
        groups.append("A_persistent_FP")

    if label == 1 and d0 == 1 and d1 == 0 and d2 == 0:
        groups.append("B_C0_correct_R1_C2_FN")

    if label == 1 and d0 == 1 and d1 == 1 and d2 == 0:
        groups.append("C_C2_only_FN")

    if label == 1 and d2 == 1 and (d0 == 0 or d1 == 0):
        groups.append("D_recovery_on_C2_fake")

    if label == 0 and d2 == 0 and (d0 == 1 or d1 == 1):
        groups.append("D_recovery_on_C2_real")

    if near_threshold_distance(row) <= NEAR_THRESHOLD_MARGIN:
        groups.append("E_near_threshold")

    return groups


def main():
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    joined_rows = build_joined_rows()

    all_cases_csv = DEST_ROOT / "all_joined_C0_R1_C2_cases.csv"
    with all_cases_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(joined_rows[0].keys()))
        writer.writeheader()
        writer.writerows(joined_rows)

    selected = []
    group_counts = {}

    for row in joined_rows:
        for group in assign_groups(row):
            key = (group, row["subset_id"])
            group_counts.setdefault(key, 0)

            if group_counts[key] >= MAX_PER_GROUP_PER_SUBSET:
                continue

            selected_row = dict(row)
            selected_row["case_group"] = group
            selected_row["score_drop_C0_to_C2"] = score_drop(row)
            selected_row["near_threshold_distance"] = near_threshold_distance(row)
            selected.append(selected_row)
            group_counts[key] += 1

    # Sort selected cases: strongest C0->C2 drops first inside copied set
    selected.sort(
        key=lambda r: (
            r["case_group"],
            r["subset_id"],
            -abs(r["score_drop_C0_to_C2"]),
            r["near_threshold_distance"],
        )
    )

    selected_csv = DEST_ROOT / "selected_error_cases.csv"

    if selected:
        with selected_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
            writer.writeheader()
            writer.writerows(selected)

    print(f"Saved joined table: {all_cases_csv}")
    print(f"Saved selected cases: {selected_csv}")
    print(f"Selected rows: {len(selected)}")

    copied_rows = []

    for index, row in enumerate(selected, start=1):
        safe_group = row["case_group"]
        safe_subset = row["subset_id"]
        safe_video = row["video_id"].replace("/", "_")

        case_dir = DEST_ROOT / safe_group / safe_subset / f"case_{index:03d}_{safe_video}"
        case_dir.mkdir(parents=True, exist_ok=True)

        copy_info = copy_case_videos(
            case_dir=case_dir,
            subset=row["subset_id"],
            label=row["label"],
            video_id=row["video_id"],
        )

        case_row = dict(row)
        case_row.update(copy_info)
        case_row["case_dir"] = str(case_dir)

        write_case_metadata(case_dir, case_row)
        copied_rows.append(case_row)

    copied_csv = DEST_ROOT / "copied_error_cases.csv"
    if copied_rows:
        with copied_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(copied_rows[0].keys()))
            writer.writeheader()
            writer.writerows(copied_rows)

    print(f"Saved copied cases metadata: {copied_csv}")

    print("\nCase counts:")
    for key in sorted(group_counts):
        print(f"{key[0]} | {key[1]}: {group_counts[key]}")


if __name__ == "__main__":
    main()