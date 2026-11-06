#!/usr/bin/env bash
set -euo pipefail

BIN=/opt/allen-cahn/bin/allen-cahn-newton-nn
VERIFY=/opt/allen-cahn/bin/verify-results
PRESET="${1:-baseline}"

case "${PRESET}" in
  smoke)
    OUTPUT=/results/smoke
    ARGS=(0.0006 "${OUTPUT}" 7 0.5 8 8 5 0.0003)
    ;;
  baseline)
    OUTPUT=/results/baseline
    ARGS=(0.03 "${OUTPUT}" 7 0.5 8 8 5 0.0003)
    ;;
  spatial-fine)
    OUTPUT=/results/spatial-fine
    ARGS=(0.03 "${OUTPUT}" 7 0.25 16 10 5 0.0003)
    ;;
  time-fine)
    OUTPUT=/results/time-fine
    ARGS=(0.03 "${OUTPUT}" 7 0.5 8 8 5 0.00015)
    ;;
  production)
    OUTPUT=/results/production
    ARGS=(0.03 "${OUTPUT}" 7 0.25 16 10 5 0.00015)
    ;;
  custom)
    shift
    if [[ $# -ne 8 ]]; then
      echo "Custom usage: custom FINAL_TIME OUTPUT MAX_ADAPT ETA_TOL INITIAL_N MAX_LEVEL INITIAL_ADAPT DT" >&2
      exit 2
    fi
    OUTPUT="$2"
    ARGS=("$@")
    ;;
  *)
    echo "Unknown preset: ${PRESET}" >&2
    echo "Choose: smoke, baseline, spatial-fine, time-fine, production, or custom" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT}"
echo "============================================================"
echo "Starting Allen-Cahn run: ${PRESET}"
echo "Results will be written to: ${OUTPUT}"
echo "This program prints one progress block per time step."
echo "============================================================"

"${BIN}" "${ARGS[@]}" 2>&1 | tee "${OUTPUT}/console.log"
"${VERIFY}" "${OUTPUT}"

