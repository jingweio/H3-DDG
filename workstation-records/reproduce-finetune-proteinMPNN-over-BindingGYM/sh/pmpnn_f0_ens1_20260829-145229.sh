#!/bin/bash
# BindingGYM official ProteinMPNN finetune -- fold 0, on the workstation A100.
#
# Moved off Ibex after eight queue readings over ~16h never advanced past "starts in ~40h"
# (they oscillated within 08-30 21:08 .. 08-31 09:18). Fold 0 needs 1-7h of actual compute, and
# the A100 has been idle since f3/f4 finished at 03:25.
#
# FROM SCRATCH, deliberately -- the Ibex resume state is NOT carried over. It would save only
# ep0's evaluation (~34 min; training is 31 s/epoch), but its best_valid_metric = 0.489961 was
# measured with the upstream 5-decoding-order ensemble. Measured on fold 3, switching to a single
# order moves the same model by -0.054, so that carried-over best would be systematically too high:
# later epochs would score "no improvement" for protocol reasons and patience-3 could stop the run
# on an artefact. Not worth 34 minutes.
#
# Knobs in main_f0_fast.py: EVAL_ENSEMBLE=1, EVAL_EVERY=3, STOP_AT=0.52.
# NOTE: STOP_AT was picked against ensemble=5 numbers (f0 reached 0.489961 there). Under a single
# decoding order the comparable value is ~0.05 lower, so 0.52 may simply never trigger -- in which
# case patience-3 decides, which is the SAME rule as folds 1-4 and therefore more consistent, not
# less. Either outcome is fine; the record must say which one fired.
set -euo pipefail

source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate bgym-official

ROOT=/home/guoj0f/repos/H3-DDG/reproduce
cd "$ROOT/bgym_official/training"
echo "[synced_commit] $(head -1 "$ROOT/.synced_commit" 2>/dev/null)"

IN=/data/guoj0f/share/BindingGYM/input
test -d "$IN/Binding_substitutions_DMS" || { echo "FATAL: shared BindingGYM data missing"; exit 1; }
test -s ./cache/v_48_020.pt            || { echo "FATAL: missing ProteinMPNN weights"; exit 1; }
test -s ./cache/BindingGYM_cluster.tsv || { echo "FATAL: missing cluster table"; exit 1; }

grep -q 'EVAL_ENSEMBLE = 1' main_f0_fast.py && grep -q 'EVAL_EVERY = 3' main_f0_fast.py \
  && grep -q 'STOP_AT = 0.52' main_f0_fast.py || {
  echo "FATAL: main_f0_fast.py is not the patched variant"; exit 1; }

# Fold 0 starts clean. Archived runs (_archived*) are exempt; live state anywhere else is not.
find . \( -name "resume_fold0.pt" -o -name "model0.ckpt" \) -not -path "*/_archived*" | grep -q . \
  && { echo "FATAL: pre-existing live state for fold 0"; exit 1; }
echo "[clean] fold 0 starts from scratch, EVAL_ENSEMBLE=1 EVAL_EVERY=3 STOP_AT=0.52"

python -c "
import torch, numpy, sklearn
n = torch.cuda.get_device_name(0); print('GPU:', n); assert 'A100' in n, n
print('numpy', numpy.__version__, '| sklearn', sklearn.__version__)
assert (numpy.__version__, sklearn.__version__) == ('1.24.4','1.3.2'), \
    'the inter-assay fold assignment depends on these -- see bgym_official/UPSTREAM.md'
"
echo "--- GPU occupancy at launch (shared machine) ---"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader

echo ""
echo "=================== fold 0 start $(date '+%F %T') ==================="
python main_f0_fast.py \
  --train_dms_mapping "$IN/BindingGYM.csv" \
  --dms_input "$IN/Binding_substitutions_DMS" \
  --structure_path "$IN/structures" \
  --model_type structure --mode inter --split cluster \
  --use_weight pretrained --batch_size 8 --seed 42 \
  --fold 0 --tmp_path inter_cluster_structure
echo "=================== fold 0 done  $(date '+%F %T') ==================="
echo "EXIT=$?"
echo "=== pmpnn f0 DONE ==="
