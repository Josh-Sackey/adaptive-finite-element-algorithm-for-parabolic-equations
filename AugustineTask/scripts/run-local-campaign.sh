#!/usr/bin/env bash
# Run the full convergence campaign on this workstation, one case at a time,
# so that recorded wall times are not distorted by competing runs.
# Usage: ./scripts/run-local-campaign.sh [case ...]
# Default cases: smoke baseline time-fine spatial-fine production
set -uo pipefail
cd "$(dirname "$0")/.."

IMAGE=allen-cahn-newton-nn:1.0
THREADS="${OMP_NUM_THREADS:-4}"
CASES=("$@")
[[ ${#CASES[@]} -eq 0 ]] && CASES=(smoke baseline time-fine spatial-fine production)

mkdir -p container-output
STATUS=container-output/campaign-status.txt
log() { echo "$(date '+%F %T')  $*" | tee -a "${STATUS}"; }

docker image inspect "${IMAGE}" >/dev/null || { log "ERROR: image ${IMAGE} not found"; exit 1; }

log "Campaign start: ${CASES[*]} (OMP_NUM_THREADS=${THREADS})"
for CASE in "${CASES[@]}"; do
  if [[ -f "container-output/${CASE}/.done" ]]; then
    log "SKIP ${CASE} (already completed)"
    continue
  fi
  rm -rf "container-output/${CASE}"
  log "START ${CASE}"
  start=$(date +%s)
  docker run --rm --name "allen-cahn-${CASE}" \
    --user "$(id -u):$(id -g)" \
    --volume "$(pwd)/container-output:/results" \
    --env OMP_NUM_THREADS="${THREADS}" \
    "${IMAGE}" "${CASE}" > "container-output/${CASE}.stdout.log" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - start ))
  if [[ ${rc} -eq 0 ]]; then
    touch "container-output/${CASE}/.done"
    log "DONE  ${CASE}  (${elapsed} s)"
  else
    log "FAIL  ${CASE}  exit=${rc}  (${elapsed} s) - see container-output/${CASE}.stdout.log"
    [[ "${CASE}" == smoke ]] && { log "Smoke test failed; stopping campaign."; exit 1; }
  fi
done
log "Campaign finished"
