#!/usr/bin/env bash
set -euo pipefail

COND="${1:?Usage: $0 C2_h264_crf40|R1_resize_x05|L1_blur_sigma1|P1_platform_like}"

SRC_ROOT="${SRC_ROOT:-$HOME/deepfake_lab/deepfake_data/FaceForensics++}"
DST_BASE="${DST_BASE:-$HOME/deepfake_lab/deepfake_data_degraded}"
DST_ROOT="$DST_BASE/$COND/FaceForensics++"

JOBS="${JOBS:-4}"
FOLLOW_SYMLINKS="${FOLLOW_SYMLINKS:-0}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is not installed or not available in PATH"
  exit 1
fi

if [[ ! -d "$SRC_ROOT" ]]; then
  echo "ERROR: source dataset root does not exist: $SRC_ROOT"
  exit 1
fi

mkdir -p "$DST_ROOT"

case "$COND" in
  C2_h264_crf40)
    CRF="40"
    FILTER="none"
    ;;

  R1_resize_x05)
    CRF="18"
    FILTER="scale=trunc(iw*0.5/2)*2:trunc(ih*0.5/2)*2,scale=trunc(iw*2/2)*2:trunc(ih*2/2)*2,setsar=1"
    ;;

  L1_blur_sigma1)
    CRF="18"
    FILTER="gblur=sigma=1.0"
    ;;

  P1_platform_like)
    CRF="35"
    FILTER="scale=trunc(iw*0.5/2)*2:trunc(ih*0.5/2)*2,scale=trunc(iw*2/2)*2:trunc(ih*2/2)*2,gblur=sigma=0.7,setsar=1"
    ;;

  *)
    echo "Unknown condition: $COND"
    exit 1
    ;;
esac

process_one() {
  local in_file="$1"
  local rel_path="${in_file#$SRC_ROOT/}"
  local out_file="$DST_ROOT/${rel_path%.*}.mp4"
  local tmp_file="${out_file}.tmp.mp4"

  mkdir -p "$(dirname "$out_file")"

  if [[ -s "$out_file" ]]; then
    echo "SKIP: $rel_path"
    return 0
  fi

  echo "PROCESS [$COND]: $rel_path"

  rm -f "$tmp_file"

  if [[ "$FILTER" == "none" ]]; then
    ffmpeg -nostdin -hide_banner -loglevel error -y \
      -i "$in_file" \
      -map 0:v:0 -an \
      -c:v libx264 -preset veryfast -crf "$CRF" \
      -pix_fmt yuv420p \
      -movflags +faststart \
      "$tmp_file"
  else
    ffmpeg -nostdin -hide_banner -loglevel error -y \
      -i "$in_file" \
      -map 0:v:0 -an \
      -vf "$FILTER" \
      -c:v libx264 -preset veryfast -crf "$CRF" \
      -pix_fmt yuv420p \
      -movflags +faststart \
      "$tmp_file"
  fi

  mv "$tmp_file" "$out_file"
}

export -f process_one
export SRC_ROOT DST_ROOT COND CRF FILTER

echo "Source:        $SRC_ROOT"
echo "Destination:   $DST_ROOT"
echo "Condition:     $COND"
echo "CRF:           $CRF"
echo "Filter:        $FILTER"
echo "Jobs:          $JOBS"
echo "Follow links:  $FOLLOW_SYMLINKS"

if [[ "$FOLLOW_SYMLINKS" == "1" ]]; then
  find -L "$SRC_ROOT" -type f -iname "*.mp4" -print0 \
    | xargs -r -0 -n 1 -P "$JOBS" bash -c 'process_one "$1"' _
else
  find "$SRC_ROOT" -type f -iname "*.mp4" -print0 \
    | xargs -r -0 -n 1 -P "$JOBS" bash -c 'process_one "$1"' _
fi

echo "Done: $COND"