#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/deepfake_lab/deepfake-diploma-benchmark"
SRC_ROOT="$HOME/deepfake_lab/deepfake_data"
DST_ROOT="$HOME/deepfake_lab/deepfake_data_ftcn16"

CONFIG="$REPO/preprocessing/config.yaml"
DATASETS_DIR="$REPO/datasets"
RGB_LINK="$DATASETS_DIR/rgb"

echo "Repository: $REPO"
echo "Source root: $SRC_ROOT"
echo "FTCN16 root: $DST_ROOT"

cd "$REPO"

if [[ ! -d "$SRC_ROOT/FaceForensics++" ]]; then
  echo "ERROR: source FaceForensics++ root not found: $SRC_ROOT/FaceForensics++"
  exit 1
fi

if [[ "$DST_ROOT" != "$HOME/deepfake_lab/deepfake_data_ftcn16" ]]; then
  echo "ERROR: unexpected DST_ROOT safety check failed: $DST_ROOT"
  exit 1
fi

echo
echo "Preparing clean FTCN16 data root with symlinked MP4 files..."
mkdir -p "$DST_ROOT"

# Recreate only the FTCN16 clean root. This does not touch the original UCF-8 data.
rm -rf "$DST_ROOT/FaceForensics++"
mkdir -p "$DST_ROOT/FaceForensics++"

find "$SRC_ROOT/FaceForensics++" \
  -mindepth 2 -maxdepth 2 \
  -type f -name "*.mp4" \
  | while read -r src; do
      rel="${src#$SRC_ROOT/}"
      dst="$DST_ROOT/$rel"
      mkdir -p "$(dirname "$dst")"
      ln -sf "$src" "$dst"
    done

echo
echo "Linked MP4 files:"
find "$DST_ROOT/FaceForensics++" -type l -name "*.mp4" | wc -l

echo
echo "Pointing datasets/rgb to FTCN16 clean root..."
mkdir -p "$DATASETS_DIR"
ln -sfn "$DST_ROOT" "$RGB_LINK"
readlink -f "$RGB_LINK"

echo
echo "Patching preprocessing/config.yaml temporarily: num_frames default -> 16"
BACKUP="$CONFIG.backup_ftcn16_$(date +%Y%m%d_%H%M%S)"
cp "$CONFIG" "$BACKUP"

restore_config() {
  if [[ -f "$BACKUP" ]]; then
    mv "$BACKUP" "$CONFIG"
    echo "Restored preprocessing/config.yaml"
  fi
}
trap restore_config EXIT

python3 - <<'PY'
import yaml
from pathlib import Path

path = Path("preprocessing/config.yaml")
with path.open() as f:
    cfg = yaml.safe_load(f)

cfg["preprocess"]["dataset_root_path"]["default"] = "../datasets/rgb"
cfg["preprocess"]["num_frames"]["default"] = 16
cfg["preprocess"]["mode"]["default"] = "fixed_num_frames"
cfg["preprocess"]["comp"]["default"] = "raw"

with path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print("preprocess.dataset_root_path.default =", cfg["preprocess"]["dataset_root_path"]["default"])
print("preprocess.num_frames.default =", cfg["preprocess"]["num_frames"]["default"])
print("preprocess.mode.default =", cfg["preprocess"]["mode"]["default"])
print("preprocess.comp.default =", cfg["preprocess"]["comp"]["default"])
PY

echo
echo "Running preprocessing for clean FTCN16 frames..."
python3 preprocessing/preprocess.py

echo
echo "Checking extracted frame counts..."
python3 - <<'PY'
from pathlib import Path
from collections import Counter

roots = [
    Path("~/deepfake_lab/deepfake_data_ftcn16/FaceForensics++/original/frames").expanduser(),
    Path("~/deepfake_lab/deepfake_data_ftcn16/FaceForensics++/Deepfakes/frames").expanduser(),
    Path("~/deepfake_lab/deepfake_data_ftcn16/FaceForensics++/Face2Face/frames").expanduser(),
    Path("~/deepfake_lab/deepfake_data_ftcn16/FaceForensics++/FaceSwap/frames").expanduser(),
    Path("~/deepfake_lab/deepfake_data_ftcn16/FaceForensics++/NeuralTextures/frames").expanduser(),
]

for root in roots:
    print("\nROOT:", root)
    if not root.exists():
        print("  missing")
        continue

    counts = Counter()
    examples = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            n = len(list(d.glob("*.png")) + list(d.glob("*.jpg")))
            counts[n] += 1
            if len(examples) < 5:
                examples.append((d.name, n))

    print("  count distribution:", counts.most_common(10))
    print("  examples:", examples)
PY

echo
echo "FTCN16 clean frame preparation finished."