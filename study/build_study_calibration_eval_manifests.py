#!/usr/bin/env python3
import argparse
import copy
import json
import os
import random
from typing import Dict, List, Tuple

FFPP_SUBSETS = ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def stable_sample(video_ids: List[str], k: int, seed: int) -> List[str]:
    rnd = random.Random(seed)
    ids = list(video_ids)
    rnd.shuffle(ids)
    return sorted(ids[:k])


def split_available_universe(
    video_dict: Dict[str, Dict],
    calibration_ratio: float,
    min_calibration_videos: int,
    seed: int,
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Split one available universe into:
      - calibration split
      - evaluation split

    Returns:
      calibration_out, evaluation_out
    """
    video_ids = sorted(video_dict.keys())
    n_total = len(video_ids)

    if n_total == 0:
        return {}, {}

    desired = max(min_calibration_videos, int(round(n_total * calibration_ratio)))

    if n_total == 1:
        n_cal = 1
    else:
        # keep at least one video for evaluation when possible
        n_cal = min(desired, n_total - 1)
        n_cal = max(1, n_cal)

    calibration_ids = set(stable_sample(video_ids, n_cal, seed))

    calibration_out = {}
    evaluation_out = {}

    for vid, info in video_dict.items():
        if vid in calibration_ids:
            calibration_out[vid] = copy.deepcopy(info)
        else:
            evaluation_out[vid] = copy.deepcopy(info)

    return calibration_out, evaluation_out


def ensure_split_and_compression(block: Dict, split_name: str, compression: str):
    if split_name not in block:
        block[split_name] = {}
    if compression not in block[split_name]:
        block[split_name][compression] = {}


def summarize_manifest(
    manifest: Dict,
    subset_name: str,
    compression: str,
    calibration_split_name: str,
    evaluation_split_name: str,
) -> Dict[str, int]:
    out = {}
    for lbl in manifest[subset_name]:
        for split in ["train", calibration_split_name, evaluation_split_name]:
            split_block = manifest[subset_name][lbl].get(split, {})
            comp_block = split_block.get(compression, {})
            out[f"{lbl}:{split}:{compression}:videos"] = len(comp_block)
            out[f"{lbl}:{split}:{compression}:frames"] = sum(
                len(v.get("frames", [])) for v in comp_block.values()
            )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="./preprocessing/dataset_json_v6",
        help="Directory with original FF++ JSON manifests",
    )
    parser.add_argument(
        "--output-dir",
        default="./preprocessing/dataset_json_study_ce",
        help="Directory to write study-specific calibration/evaluation manifests",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=FFPP_SUBSETS,
        help="FF++ subsets to process",
    )
    parser.add_argument(
        "--compression",
        default="raw",
        help="Compression key to operate on (default: raw)",
    )
    parser.add_argument(
        "--source-split",
        default="test",
        choices=["train", "val", "test"],
        help="Existing split to cut into calibration/evaluation",
    )
    parser.add_argument(
        "--calibration-split-name",
        default="val",
        help="Target split name for calibration rows",
    )
    parser.add_argument(
        "--evaluation-split-name",
        default="test",
        help="Target split name for held-out evaluation rows",
    )
    parser.add_argument(
        "--calibration-ratio",
        type=float,
        default=0.15,
        help="Fraction of available source universe assigned to calibration",
    )
    parser.add_argument(
        "--min-calibration-videos",
        type=int,
        default=50,
        help="Minimum number of calibration videos per label per subset if available",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="Random seed for reproducible split creation",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Study calibration/evaluation manifests created.")
    print(f"input_dir:  {args.input_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"compression: {args.compression}")
    print(f"source_split: {args.source_split}")
    print(f"calibration_split_name: {args.calibration_split_name}")
    print(f"evaluation_split_name: {args.evaluation_split_name}")
    print(f"calibration_ratio: {args.calibration_ratio}")
    print(f"min_calibration_videos: {args.min_calibration_videos}")
    print("")

    for subset in args.subsets:
        in_path = os.path.join(args.input_dir, f"{subset}.json")
        data = load_json(in_path)

        if subset not in data:
            raise ValueError(f"Top-level subset key {subset} missing in {in_path}")

        new_data = copy.deepcopy(data)
        subset_block = new_data[subset]

        for lbl in subset_block:
            ensure_split_and_compression(subset_block[lbl], args.calibration_split_name, args.compression)
            ensure_split_and_compression(subset_block[lbl], args.evaluation_split_name, args.compression)

            source_block = subset_block[lbl].get(args.source_split, {}).get(args.compression, {})

            calibration_out, evaluation_out = split_available_universe(
                video_dict=source_block,
                calibration_ratio=args.calibration_ratio,
                min_calibration_videos=args.min_calibration_videos,
                seed=args.seed + hash((subset, lbl)) % 100000,
            )

            subset_block[lbl][args.calibration_split_name][args.compression] = calibration_out
            subset_block[lbl][args.evaluation_split_name][args.compression] = evaluation_out

        out_path = os.path.join(args.output_dir, f"{subset}.json")
        save_json(out_path, new_data)

        summary = summarize_manifest(
            new_data,
            subset_name=subset,
            compression=args.compression,
            calibration_split_name=args.calibration_split_name,
            evaluation_split_name=args.evaluation_split_name,
        )

        print(f"[{subset}] -> {out_path}")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("")


if __name__ == "__main__":
    main()