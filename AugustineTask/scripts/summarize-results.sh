#!/usr/bin/env bash
# Print a compact summary of every completed case in container-output/.
cd "$(dirname "$0")/../container-output" || exit 1
[[ -f campaign-status.txt ]] && { echo "--- campaign status ---"; cat campaign-status.txt; echo; }
for c in smoke baseline time-fine spatial-fine production; do
  [[ -f "$c/console.log" ]] || continue
  echo "=== $c  (VTU files: $(ls "$c"/*.vtu 2>/dev/null | wc -l))"
  grep -A6 'RUN COMPLETED' "$c/console.log" | sed -n '2,5p'
  if [[ -f "$c/simulation-history.csv" ]]; then
    head -1 "$c/simulation-history.csv"
    sed -n '2p;$p' "$c/simulation-history.csv"
  fi
  [[ -f "$c/neural-training.csv" ]] && echo "NN last: $(tail -1 "$c/neural-training.csv")"
  echo
done
