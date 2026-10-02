#!/usr/bin/env bash
set -euo pipefail

top_fraction="${1:-0.6}"
final_time="${2:-0.05}"
output_root="${3:-runs/comparison}"

mkdir -p "${output_root}"
: > "${output_root}/run.log"
for mode in classical nn-once-per-time nn-every-crossing; do
  ./build/transfer-strategy-comparison "${mode}" "${top_fraction}" \
    "${final_time}" "${output_root}" 2>&1 | tee -a "${output_root}/run.log"
done
python3 compare_results.py "${output_root}"
