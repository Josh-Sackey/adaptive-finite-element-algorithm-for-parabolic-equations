#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:?Usage: verify-results OUTPUT_DIRECTORY}"
required=(allen-cahn.pvd simulation-history.csv newton-history.csv neural-training.csv)
for file in "${required[@]}"; do
  if [[ ! -s "${OUTPUT}/${file}" ]]; then
    echo "ERROR: missing or empty ${OUTPUT}/${file}" >&2
    exit 1
  fi
done

VTU_COUNT=$(find "${OUTPUT}" -maxdepth 1 -name 'allen-cahn-*.vtu' -type f | wc -l)
LAST_ROW=$(tail -n 1 "${OUTPUT}/simulation-history.csv")
FINAL_TIME=$(printf '%s\n' "${LAST_ROW}" | cut -d, -f2)
FINAL_DOFS=$(printf '%s\n' "${LAST_ROW}" | cut -d, -f5)
ELAPSED=$(printf '%s\n' "${LAST_ROW}" | cut -d, -f12)

echo
echo "==================== RUN COMPLETED ===================="
echo "VTU snapshots : ${VTU_COUNT}"
echo "Final time    : ${FINAL_TIME}"
echo "Final DoFs    : ${FINAL_DOFS}"
echo "Elapsed time  : ${ELAPSED} seconds"
echo "ParaView file : ${OUTPUT}/allen-cahn.pvd"
echo "Summary CSV   : ${OUTPUT}/simulation-history.csv"
echo "======================================================="
