#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: Docker is not installed or is not running." >&2
  echo "Install Docker, then run this script again." >&2
  exit 1
fi

JOBS="${BUILD_JOBS:-4}"
echo "Building the Docker image. The first build may take 30-90 minutes."
docker build --build-arg BUILD_JOBS="${JOBS}" \
  --tag allen-cahn-newton-nn:1.0 .
echo "Docker image ready: allen-cahn-newton-nn:1.0"

