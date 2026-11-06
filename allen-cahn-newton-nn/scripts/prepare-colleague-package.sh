#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f allen-cahn-newton-nn.sif ]]; then
  echo "ERROR: allen-cahn-newton-nn.sif does not exist." >&2
  echo "Create it first with ./scripts/build-apptainer.sh" >&2
  exit 1
fi

PACKAGE=allen-cahn-colleague-package
rm -rf "${PACKAGE}"
mkdir -p "${PACKAGE}/scripts" "${PACKAGE}/docs"
cp allen-cahn-newton-nn.sif "${PACKAGE}/"
cp scripts/run-slurm.sh scripts/run-pbs.sh "${PACKAGE}/scripts/"
cp docs/HPC-CONTAINER-GUIDE.md "${PACKAGE}/docs/"
(cd "${PACKAGE}" && sha256sum allen-cahn-newton-nn.sif > SHA256SUMS)
tar -czf "${PACKAGE}.tar.gz" "${PACKAGE}"

echo "Colleague package ready: $(pwd)/${PACKAGE}.tar.gz"
echo "The recipient can verify it with: sha256sum -c SHA256SUMS"

