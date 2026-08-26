#!/bin/bash
# Usage: ./submit_finetune.sh 0 1 2 3 4      (or any subset)
#
# Per-fold walltimes.
#
# Two measurements, not extrapolation -- the first estimate here was a GPU-only benchmark and was
# wrong by ~10x, so these come from real runs:
#   f2 smoke (50852061):  ~80 min startup + 6.7 min/epoch  (training 26 s, eval 6:16 at 78 rows/s)
#   f3 measure (50878825): 1 epoch inside 3:30             (142,905 rows, structures to 1041 res)
# The ~80 min startup is now cached away (PATCHES.md #3), and the f3 figure was measured BEFORE
# that cache existed, so it is an upper bound.
#
# Cost is dominated by scoring the full held-out fold every epoch, which patience-3 requires, and
# the folds differ ~7x because f0/f3 hold the large structures. Sized so a patience-3 stop around
# epoch 20 fits; --resume (PATCHES.md #4) covers the case where it does not, which is why f0/f3
# can take a 7h slot rather than needing a walltime that would sit pending for days.
declare -A WALL=( [0]=07:00:00 [1]=04:00:00 [2]=04:00:00 [3]=07:00:00 [4]=05:00:00 )

cd "$(dirname "$0")"
for F in "$@"; do
  JID=$(sbatch --parsable --job-name="pmpnn_f${F}" --time="${WALL[$F]}" \
        finetune_proteinmpnn_inter.sh "${F}")
  echo "fold ${F}: job ${JID}  (walltime ${WALL[$F]})"
done
