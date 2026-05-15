import argparse
import json
from pathlib import Path


def replace_in_obj(obj, old: str, new: str):
    if isinstance(obj, str):
        return obj.replace(old, new)

    if isinstance(obj, list):
        return [replace_in_obj(item, old, new) for item in obj]

    if isinstance(obj, dict):
        return {
            key: replace_in_obj(value, old, new)
            for key, value in obj.items()
        }

    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-json-dir", required=True)
    parser.add_argument("--dst-json-dir", required=True)
    parser.add_argument("--from-token", required=True)
    parser.add_argument("--to-token", required=True)
    args = parser.parse_args()

    src_dir = Path(args.src_json_dir)
    dst_dir = Path(args.dst_json_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source JSON folder does not exist: {src_dir}")

    json_files = sorted(src_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    for src_file in json_files:
        with src_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        replaced = replace_in_obj(data, args.from_token, args.to_token)

        dst_file = dst_dir / src_file.name

        with dst_file.open("w", encoding="utf-8") as file:
            json.dump(replaced, file, ensure_ascii=False, indent=2)

        print(f"Saved: {dst_file}")


if __name__ == "__main__":
    main()