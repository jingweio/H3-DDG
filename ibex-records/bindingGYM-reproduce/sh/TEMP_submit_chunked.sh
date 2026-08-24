#!/bin/bash
# TEMPORARY submitter for TEMP_bindinggym_chunked_20260824.sh -- see that file's header.
# Usage:  ./TEMP_submit_chunked.sh 0 3        (submits folds 0 and 3)
#
# Walltime 7h, from MEASURED cost, not the old K^3 extrapolation:
#   f1  8h requested -> 2:47:18 actual   (35%)
#   f2  8h requested -> 2:31:51 actual   (32%)
#   f4 10h requested -> 4:05:46 actual   (41%)
# Max observed 4:05:46; 7h is that +71%. f0/f3 have more eval rows (114k/143k vs 35-55k) but the
# 20,000 training iterations are identical across folds, so the extra cost is eval-only and the
# spread above is already mostly eval.  7h will most likely finish in ONE chunk; --resume is the
# safety net if it does not, which is the whole point of keeping the walltime this short.
WALL=07:00:00

cd "$(dirname "$0")"
for F in "$@"; do
  JID=$(sbatch --parsable \
        --job-name="bgym_f${F}c" \
        --time="${WALL}" \
        TEMP_bindinggym_chunked_20260824.sh "${F}")
  echo "fold ${F}: job ${JID}  (walltime ${WALL}, chunked/resumable)"
done
