#!/usr/bin/env bash
# Re-run the baseline case with 1 and 4 LibTorch threads to measure
# run-to-run reproducibility. Results: container-output/repro-baseline-threads{1,4}
set -euo pipefail
cd "$(dirname "$0")/.."
for t in 1 4; do
  out="repro-baseline-threads${t}"
  rm -rf "container-output/${out}"
  docker run --rm --user "$(id -u):$(id -g)" \
    --volume "$(pwd)/container-output:/results" \
    --env OMP_NUM_THREADS="${t}" \
    allen-cahn-newton-nn:1.0 custom 0.03 "/results/${out}" 7 0.5 8 8 5 0.0003 \
    > "container-output/${out}.stdout.log" 2>&1
  echo "threads=${t}: $(tail -1 "container-output/${out}/simulation-history.csv")"
done
echo "original : $(tail -1 container-output/baseline/simulation-history.csv)"
