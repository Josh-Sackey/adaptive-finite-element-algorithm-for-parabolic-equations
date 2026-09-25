#!/usr/bin/env bash
set -euo pipefail

top_fraction="${1:-0.6}"
final_time="${2:-1.0}"
output_root="${3:-runs/comparison}"

mkdir -p "${output_root}"
./build/fig8-quads-nn both "${top_fraction}" "${final_time}" "${output_root}" \
  2>&1 | tee "${output_root}/run.log"
python3 compare_results.py "${output_root}"
