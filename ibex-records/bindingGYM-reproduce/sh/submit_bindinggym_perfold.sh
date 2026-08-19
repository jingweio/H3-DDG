#!/bin/bash
# Submit the 5 BindingGYM inter-assay folds as INDEPENDENT jobs with per-fold walltimes.
#
# Walltime table. Estimates come from the measured a100 anchor in the record md (§5.4):
# eval cost is dominated by the 3-body hypergraph attention over (K,K) hyperedge pairs, K = L/4,
# so it tracks structure size, not mutation depth. Training adds ~4h to every fold.
#   fold0  eval ~10.9h (1HE8 K=228)  + train ~4h  => ~15h   -> 18h
#   fold1  eval  ~1.2h (K~28, measured directly)  + ~4h  => ~5.3h -> 8h
#   fold2  eval  ~1.4h (6M17 K=232)  + ~4h  => ~5.5h -> 8h
#   fold3  eval  ~9.3h (6M0J K=197)  + ~4h  => ~13.4h -> 16h
#   fold4  eval  ~2.9h (4ZFF K=130)  + ~4h  => ~7h   -> 10h
# Every value stays under gpu24's 24h limit, which keeps the largest a100 pool (32 nodes) eligible.
# Ibex also grants exactly 1h of grace past the limit, and periodic checkpoints make an overrun
# recoverable rather than fatal -- so these are deliberately tighter than the old uniform 23h.
declare -A WALL=( [0]=18:00:00 [1]=08:00:00 [2]=08:00:00 [3]=16:00:00 [4]=10:00:00 )

cd "$(dirname "$0")"
for F in 0 1 2 3 4; do
  JID=$(sbatch --parsable \
        --job-name="bgym_f${F}" \
        --time="${WALL[$F]}" \
        bindinggym_perfold_20260819.sh "${F}")
  echo "fold ${F}: job ${JID}  (walltime ${WALL[$F]})"
done
