#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: Docker is not installed or is not running." >&2
  exit 1
fi
PRESET="${1:-baseline}"
mkdir -p container-output
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --volume "$(pwd)/container-output:/results" \
  --env OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
  allen-cahn-newton-nn:1.0 "${PRESET}"

