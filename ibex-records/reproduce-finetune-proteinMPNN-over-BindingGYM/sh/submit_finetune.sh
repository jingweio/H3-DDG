#!/bin/bash
# Usage: ./submit_finetune.sh 0 1 2 3 4      (or any subset)
#
# Per-fold walltimes, from measured per-epoch cost (A4500, batch 8; a100 expected ~2x faster,
# unmeasured). The cost is dominated by evaluating the held-out fold every epoch, and the folds
# differ ~7x because f0/f3 hold the large structures:
#     f0 21.9 min/epoch   f1 3.3   f2 3.1   f3 22.2   f4 8.5
# Sized so patience-3 stopping around epoch 20 finishes comfortably, with --resume unavailable
# upstream, so a fold that runs long must be re-run rather than continued -- hence the headroom
# on f0/f3. Short walltimes matter here: 7h has started in ~11h, 16h+ has sat for 5 days.
declare -A WALL=( [0]=07:00:00 [1]=03:00:00 [2]=03:00:00 [3]=07:00:00 [4]=05:00:00 )

cd "$(dirname "$0")"
for F in "$@"; do
  JID=$(sbatch --parsable --job-name="pmpnn_f${F}" --time="${WALL[$F]}" \
        finetune_proteinmpnn_inter.sh "${F}")
  echo "fold ${F}: job ${JID}  (walltime ${WALL[$F]})"
done
