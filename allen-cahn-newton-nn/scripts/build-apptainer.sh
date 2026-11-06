#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: Apptainer is not installed or has not been loaded as a module." >&2
  echo "On a cluster, try: module load apptainer" >&2
  exit 1
fi

if [[ ! -f allen-cahn-newton-nn.tar ]]; then
  echo "Docker archive not found; creating it now."
  ./scripts/export-docker.sh
fi

echo "Converting the Docker archive to an Apptainer SIF image."
apptainer build --force allen-cahn-newton-nn.sif \
  docker-archive:allen-cahn-newton-nn.tar
echo "Apptainer image ready: $(pwd)/allen-cahn-newton-nn.sif"

