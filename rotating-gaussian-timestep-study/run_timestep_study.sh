#!/usr/bin/env bash
set -euo pipefail

top_fraction="${1:-0.6}"
final_time="${2:-0.05}"
output_root="${3:-runs/early-time}"
time_steps=(0.01 0.005 0.0025)

mkdir -p "${output_root}"
: > "${output_root}/run.log"

for dt in "${time_steps[@]}"; do
  dt_slug="dt-$(printf '%0.5f' "${dt}" | tr '.' 'p')"
  for mode in classical nn-once-per-time; do
    echo "=== mode=${mode} dt=${dt} T=${final_time} ===" | tee -a "${output_root}/run.log"
    ./build/rotating-gaussian-timestep-study "${mode}" "${top_fraction}" \
      "${final_time}" "${dt}" "${output_root}/${dt_slug}" \
      2>&1 | tee -a "${output_root}/run.log"
  done
done

python3 analyze_timestep_study.py "${output_root}" "${time_steps[@]}"
