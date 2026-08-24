#!/bin/bash
# Submit the untrained baseline for the two candidate paper folds.
#
# Walltime from the measured a100 anchor: f1's full job was 2:47:18 of which ~1:47 was the 20,000
# training iterations, leaving ~1:00 for 55,081 eval rows.  Eval-only therefore costs about
#   f1  55,081 rows -> ~1.0h  -> 2:30 requested
#   f3 142,905 rows -> ~2.6h  -> 5:00 requested
# Kept deliberately short: 8h jobs queued ~20h here while 16-18h jobs sat for 5+ days.
declare -A WALL=( [1]=02:30:00 [3]=05:00:00 )

cd "$(dirname "$0")"
for F in "$@"; do
  JID=$(sbatch --parsable \
        --job-name="bgym_untr_f${F}" \
        --time="${WALL[$F]}" \
        bindinggym_untrained_baseline_20260824.sh "${F}")
  echo "fold ${F}: job ${JID}  (walltime ${WALL[$F]}, eval-only)"
done
