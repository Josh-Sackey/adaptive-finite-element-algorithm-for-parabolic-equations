#!/usr/bin/env bash
#SBATCH --job-name=allen-cahn
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/allen-cahn-%A_%a.out
#SBATCH --error=logs/allen-cahn-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results

CASES=(baseline spatial-fine time-fine production)
CASE="${CASES[$SLURM_ARRAY_TASK_ID]}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Running case: ${CASE}"
apptainer run --cleanenv \
  --bind "${PWD}/results:/results" \
  --env OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  allen-cahn-newton-nn.sif "${CASE}"

