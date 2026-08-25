#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --output=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/official_strategy_20260824_%x_%j.out
#SBATCH --error=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/official_strategy_20260824_%x_%j.err

# H3-DDG model, BindingGYM's OWN inter-assay training strategy (within-assay batches + listMLE +
# uniform assay sampling + AdamW/OneCycleLR + early stopping on per-DMS Spearman).
#
# The A.4-faithful arm (sh/bindinggym_perfold_20260819.sh) collapses: output goes near-constant
# inside 5,000 iterations and the 14-assay mean Spearman is -0.0052.  BindingGYM's own published
# result for a pretrained ProteinMPNN fine-tuned under this exact split is 0.4217 ALL Spearman
# (results/ProteinMPNN_finetune_inter_cluster_metric.csv), so the strategy demonstrably works on
# this data with this backbone.  Per-fold official targets, from that same file:
#     f0 0.5542   f1 0.2719   f2 0.3035   f3 0.5550   f4 0.3916
# A run that lands near its fold's target confirms the strategy is the fix; one that lands near
# zero says something else is wrong.
#
# Memory, measured on the 1016-residue 4D5_HER2 structure (the largest training structure for
# folds 0 and 2): slate 1 peaks at 11.00 GiB, slate 2 needs ~22 GiB. A 40 GB a100 clears slate 2
# with 1.8x margin; the 20 GB A4500 does not, which is why this cannot be validated locally.
# The same figures make A.4's "batch size of 1, 2, depending on GPU memory and graph size"
# concrete: on the authors' 24 GB RTX 4090, slate 2 is exactly where it stops fitting.
#
# Resumable: --resume continues from the last completed epoch.

set -euo pipefail

FOLD=$1
# Which config: strategy (default) isolates BindingGYM's TRAINING STRATEGY while holding
# H3-DDG's architecture AND training parameters fixed. full_recipe additionally swaps in
# BindingGYM's optimiser (AdamW 1e-3, wd 0.05, OneCycleLR) -- their published 0.4217 setup, but
# then a difference cannot be attributed to the strategy alone.
ARM=${2:-strategy}
case "$ARM" in
  strategy)    CFG=./config/train_h3-ddg_bindinggym_strategy.json ;;
  full_recipe) CFG=./config/train_h3-ddg_bindinggym_official.json ;;
  bgymfull)    CFG=./config/train_h3-ddg_bindinggym_bgymfull.json ;;
  *) echo "FATAL: unknown arm '$ARM' (want strategy|full_recipe|bgymfull)"; exit 1 ;;
esac

source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate h3ddg-reproduce
cd /ibex/user/guoj0f/H3-DDG/reproduce

SAVE_DIR=./results/bindinggym_${ARM}_fold${FOLD}

echo "=== arm ${ARM} (${CFG}) | fold ${FOLD} | job ${SLURM_JOB_ID} ==="
python - <<'PY'
import torch, numpy, pandas, sklearn, Bio
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0))
print('numpy', numpy.__version__, '| sklearn', sklearn.__version__, '| biopython', Bio.__version__)
print('gpu mem (GiB)', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
assert sklearn.__version__ in ('1.2.1', '1.3.2'), 'fold split is sklearn-version sensitive'
PY

for f in ./data/BindingGYM_cache/entries.pkl ./data/BindingGYM_cache/structures.pkl; do
  test -s "$f" || { echo "FATAL: missing cache $f"; exit 1; }
done

export WANDB_MODE=disabled
srun python train_bindinggym_official.py \
  --config_path ${CFG} \
  --test_fold "${FOLD}" \
  --save_dir "${SAVE_DIR}" \
  --resume \
  --num_workers 12

echo "=== fold ${FOLD} arm ${ARM} DONE ==="
