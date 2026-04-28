#!/usr/bin/env python3
import argparse
import copy
import json
import os
import random
from typing import Dict, List, Tuple


FFPP_SUBSETS = ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]
REAL_LABEL = "FF-real"


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


def split_train_into_train_val(
    video_dict: Dict[str, Dict],
    val_ratio: float,
    min_val_videos: int,
    seed: int,
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    video_ids = sorted(video_dict.keys())
    n_total = len(video_ids)

    if n_total == 0:
        return {}, {}

    n_val = max(min_val_videos, int(round(n_total * val_ratio)))
    n_val = min(n_val, n_total)

    val_ids = set(stable_sample(video_ids, n_val, seed))
    train_out = {}
    val_out = {}

    for vid, info in video_dict.items():
        if vid in val_ids:
            val_out[vid] = copy.deepcopy(info)
        else:
            train_out[vid] = copy.deepcopy(info)

    return train_out, val_out


def dedupe_real_videos_across_subsets(
    manifests: Dict[str, Dict],
    compression: str,
    keep_subset_for_real: str = "FF-DF",
):
    """
    Keep each real video in exactly one subset's val/raw to avoid duplication across subsets.
    By default, real val videos are kept in FF-DF and removed from the other 3 subsets.
    """
    if keep_subset_for_real not in manifests:
        raise ValueError(f"keep_subset_for_real={keep_subset_for_real} not in manifests")

    keeper_real = manifests[keep_subset_for_real][keep_subset_for_real][REAL_LABEL]["val"][compression]
    keeper_ids = set(keeper_real.keys())

    for subset in manifests:
        if subset == keep_subset_for_real:
            continue
        real_val = manifests[subset][subset][REAL_LABEL]["val"][compression]
        remove_ids = [vid for vid in real_val.keys() if vid in keeper_ids]
        for vid in remove_ids:
            del real_val[vid]


def summarize_manifest(manifest: Dict, subset_name: str, compression: str) -> Dict[str, int]:
    out = {}
    for lbl in manifest[subset_name]:
        for split in ["train", "val", "test"]:
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
        default="./preprocessing/dataset_json_study_val",
        help="Directory to write new study validation manifests",
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
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of train/raw videos to move into val/raw",
    )
    parser.add_argument(
        "--min-val-videos",
        type=int,
        default=50,
        help="Minimum number of validation videos per class per subset if available",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="Random seed for reproducible split creation",
    )
    parser.add_argument(
        "--keep-real-in-subset",
        default="FF-DF",
        choices=FFPP_SUBSETS,
        help="Subset that keeps the deduplicated real validation videos",
    )
    args = parser.parse_args()

    manifests = {}

    for subset in args.subsets:
        in_path = os.path.join(args.input_dir, f"{subset}.json")
        data = load_json(in_path)

        if subset not in data:
            raise ValueError(f"Top-level subset key {subset} missing in {in_path}")

        new_data = copy.deepcopy(data)

        subset_block = new_data[subset]
        for lbl in subset_block:
            train_block = subset_block[lbl]["train"][args.compression]

            train_out, val_out = split_train_into_train_val(
                video_dict=train_block,
                val_ratio=args.val_ratio,
                min_val_videos=args.min_val_videos,
                seed=args.seed + hash((subset, lbl)) % 100000,
            )

            subset_block[lbl]["train"][args.compression] = train_out
            subset_block[lbl]["val"][args.compression] = val_out

        manifests[subset] = new_data

    # Deduplicate real validation videos across the 4 subset manifests.
    dedupe_real_videos_across_subsets(
        manifests=manifests,
        compression=args.compression,
        keep_subset_for_real=args.keep_real_in_subset,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print("Study validation manifests created.")
    print(f"input_dir:  {args.input_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"compression: {args.compression}")
    print(f"val_ratio: {args.val_ratio}")
    print(f"min_val_videos: {args.min_val_videos}")
    print(f"keep_real_in_subset: {args.keep_real_in_subset}")
    print("")

    for subset in args.subsets:
        out_path = os.path.join(args.output_dir, f"{subset}.json")
        save_json(out_path, manifests[subset])

        summary = summarize_manifest(manifests[subset], subset, args.compression)
        print(f"[{subset}] -> {out_path}")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("")


if __name__ == "__main__":
    main()