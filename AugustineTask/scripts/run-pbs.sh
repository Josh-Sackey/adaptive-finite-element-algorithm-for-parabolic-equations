#!/usr/bin/env bash
#PBS -N allen-cahn-production
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=12:00:00
#PBS -j oe

set -euo pipefail
cd "${PBS_O_WORKDIR}"
mkdir -p results
export OMP_NUM_THREADS="${NCPUS:-8}"

apptainer run --cleanenv \
  --bind "${PWD}/results:/results" \
  --env OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  allen-cahn-newton-nn.sif production

