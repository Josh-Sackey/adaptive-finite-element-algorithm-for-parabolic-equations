#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: Apptainer is not installed or has not been loaded as a module." >&2
  exit 1
fi
PRESET="${1:-baseline}"
mkdir -p container-output
apptainer run --cleanenv \
  --bind "$(pwd)/container-output:/results" \
  --env OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
  allen-cahn-newton-nn.sif "${PRESET}"

