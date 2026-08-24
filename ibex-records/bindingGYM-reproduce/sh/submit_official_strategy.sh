#!/bin/bash
# Usage: ./submit_official_strategy.sh 0 2
#
# 7h for every fold. Training is fold-independent (256 steps/epoch, <=40 epochs, patience-3 early
# stopping, which normally fires around epoch 10-20 => 1-2.5h). The fold-dependent part is the ONE
# full evaluation at the end, at the measured a100 rate of ~55k rows/hour:
#     f0 114,341 rows ~2.1h   f1 55,081 ~1.0h   f2 29,332 ~0.5h
#     f3 142,905 ~2.6h        f4 34,787 ~0.6h
# Short on purpose: 8h jobs queued ~20h on this account while 16-18h jobs sat for 5+ days.
WALL=07:00:00

cd "$(dirname "$0")"
for F in "$@"; do
  JID=$(sbatch --parsable --job-name="bgym_off_f${F}" --time="${WALL}" \
        bindinggym_official_strategy_20260824.sh "${F}")
  echo "fold ${F}: job ${JID}  (walltime ${WALL}, BindingGYM official strategy)"
done
