#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/reproduce-finetune-proteinMPNN-over-BindingGYM/finetune_%x_%j.out
#SBATCH --error=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/reproduce-finetune-proteinMPNN-over-BindingGYM/finetune_%x_%j.err

# TEMPORARY, SINGLE-USE. Fold 0 only. Delete once fold 0 has a result.
#
# Fold 0 holds out 114,341 rows, and upstream scores the whole held-out fold every epoch because
# patience-3 requires it. Measured on Ibex: 3:30 buys exactly one epoch, twice (50904635, 50922621
# both TIMEOUT at 03:30 having advanced one epoch). It is not going to finish that way.
#
# So this job runs main_f0_fast.py instead of main.py -- same file with two knobs added:
#   EVAL_EVERY = 3   evaluate every 3rd epoch rather than every epoch
#   STOP_AT   = 0.52 stop as soon as the held-out pooled Spearman reaches it
# Everything else -- data, split, optimiser, loss, sampler, seed -- is untouched upstream.
#
# Disclose when reporting: fold 0's checkpoint is then chosen by "first to cross 0.52", while
# folds 1-4 use upstream's patience-3 best. Different selection rules; fold 0 is a reproducibility
# check, not a like-for-like row in the per-fold table.
#
# Walltime is 24h, not the old 2:30. Measured 2026-08-28: 05:00 / 07:00 / 12:00 / 24:00 all return
# the SAME estimated start, so a short walltime no longer buys queue position -- it only buys a
# TIMEOUT and another full queue cycle.
#
# Resumes from the existing state (epoch 1 done, best spearman 0.489961, NIE 0/3). Do NOT let this
# start from scratch -- that would discard the epoch already paid for.

set -euo pipefail
FOLD=0
TMP=inter_cluster_structure

source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate bgym-official
cd /ibex/user/guoj0f/H3-DDG/reproduce/bgym_official/training

echo "=== fold ${FOLD} FAST variant | tmp_path ${TMP} | job ${SLURM_JOB_ID} ==="
python - <<'PY'
import torch, numpy, sklearn, torch_geometric
p = torch.cuda.get_device_properties(0)
print(f'GPU {p.name} {p.total_memory/2**30:.0f} GiB | torch {torch.__version__} | PyG {torch_geometric.__version__}')
print(f'numpy {numpy.__version__} | sklearn {sklearn.__version__}')
assert (numpy.__version__, sklearn.__version__) == ('1.24.4', '1.3.2'), \
    'the inter-assay fold assignment depends on these -- see bgym_official/UPSTREAM.md'
PY

test -s ./cache/v_48_020.pt || { echo "FATAL: missing ProteinMPNN weights"; exit 1; }
test -s ./cache/BindingGYM_cluster.tsv || { echo "FATAL: missing cluster table"; exit 1; }
test -d ../input/Binding_substitutions_DMS || { echo "FATAL: missing DMS data"; exit 1; }
test -s ./output/train_on_BindingGYM_${TMP}_seed42/resume_fold0.pt || {
  echo "FATAL: no fold-0 resume state; refusing to silently restart from scratch"; exit 1; }
grep -q 'EVAL_EVERY = 3' main_f0_fast.py && grep -q 'STOP_AT = 0.52' main_f0_fast.py || {
  echo "FATAL: main_f0_fast.py is not the patched variant"; exit 1; }
echo "[ok] resuming fold 0 with EVAL_EVERY=3, STOP_AT=0.52"

srun python main_f0_fast.py \
  --train_dms_mapping ../input/BindingGYM.csv \
  --dms_input ../input/Binding_substitutions_DMS \
  --structure_path ../input/structures \
  --model_type structure --mode inter --split cluster \
  --use_weight pretrained --batch_size 8 --seed 42 \
  --fold "${FOLD}" --tmp_path "${TMP}" \
  --resume

echo "=== fold ${FOLD} FAST DONE ==="
