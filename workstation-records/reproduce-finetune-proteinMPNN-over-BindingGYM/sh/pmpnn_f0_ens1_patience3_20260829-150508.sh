#!/bin/bash
# BindingGYM official ProteinMPNN finetune -- fold 0, workstation A100.
#
# Runs main_ens1.py -- the SAME file folds 3 and 4 used: upstream main.py with one change,
# EVAL_ENSEMBLE=1. No EVAL_EVERY, no STOP_AT. Evaluation happens every epoch and the run stops
# on upstream's patience-3, exactly as folds 1-4 did.
#
# Why this replaces the earlier EVAL_EVERY=3 / STOP_AT=0.52 variant: the speedup we actually
# needed came from the ensemble, not from skipping evaluations. Measured on this A100 --
#   upstream (ensemble 5, eval every epoch): 5 x 33 min + 31 s ~= 2.8 h per epoch
#   ensemble 1, eval every epoch:            33 min + 31 s     ~= 34 min per epoch
#   ensemble 1 + EVAL_EVERY=3:                                 ~= 11.6 min per epoch amortised
# The 5x is the ensemble; EVAL_EVERY was a further 3x on top. Giving it back costs ~6-8.5 h
# (patience-3 stopped folds 1-4 at epochs 14/10/7/4) and buys uniform model selection: all five
# folds now stop on patience-3 over a per-epoch evaluation. The only remaining protocol
# difference in the whole table is ensemble 1 (folds 0/3/4) vs 5 (folds 1/2).
#
# From scratch. The Ibex resume state is deliberately not used -- its best_valid_metric 0.489961
# was measured under the 5-order ensemble, and on fold 3 that switch moved the same model by
# -0.054, so carrying it in would make later epochs look like non-improvements for protocol
# reasons and let patience-3 stop on an artefact.
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

# Must be the ensemble-only variant: EVAL_ENSEMBLE present, the fold-0 fast knobs absent.
grep -q 'EVAL_ENSEMBLE = 1' main_ens1.py || { echo "FATAL: main_ens1.py lacks EVAL_ENSEMBLE=1"; exit 1; }
grep -qE '^(EVAL_EVERY|STOP_AT) =' main_ens1.py && { echo "FATAL: main_ens1.py carries fold-0 fast knobs"; exit 1; }
echo "[ok] main_ens1.py: EVAL_ENSEMBLE=1, per-epoch eval, upstream patience-3"

find . \( -name "resume_fold0.pt" -o -name "model0.ckpt" \) -not -path "*/_archived*" | grep -q . \
  && { echo "FATAL: pre-existing live state for fold 0"; exit 1; }
echo "[clean] fold 0 starts from scratch"

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
python main_ens1.py \
  --train_dms_mapping "$IN/BindingGYM.csv" \
  --dms_input "$IN/Binding_substitutions_DMS" \
  --structure_path "$IN/structures" \
  --model_type structure --mode inter --split cluster \
  --use_weight pretrained --batch_size 8 --seed 42 \
  --fold 0 --tmp_path inter_cluster_structure
echo "=================== fold 0 done  $(date '+%F %T') ==================="
echo "EXIT=$?"
echo "=== pmpnn f0 DONE ==="
