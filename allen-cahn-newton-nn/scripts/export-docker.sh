#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: Docker is not installed or is not running." >&2
  exit 1
fi
docker image inspect allen-cahn-newton-nn:1.0 >/dev/null
docker save --output allen-cahn-newton-nn.tar allen-cahn-newton-nn:1.0
echo "Created $(pwd)/allen-cahn-newton-nn.tar"

