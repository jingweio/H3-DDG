#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --output=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/untrained_baseline_20260824_%x_%j.out
#SBATCH --error=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/untrained_baseline_20260824_%x_%j.err

# Untrained baseline: pretrained ProteinMPNN + untrained H3-DDG heads, evaluated on a held-out
# fold with NO BindingGYM training at all.  Diagnostic, not a reproduction run.
#
# Why: the trained model's per-assay correlations are indistinguishable from zero (7 positive /
# 7 negative, mean +0.0180) and its output has collapsed to near-constant.  The only untrained
# numbers we have are 1,000-row monitoring subsamples at iteration 1, where f1 read +0.3008 --
# suspiciously close to the paper's 0.3057, and 3x the paper's own ProteinMPNN baseline row
# (0.0998).  A full-fold untrained number settles whether the pipeline can produce signal at all.
#
# f1 and f3 are the two candidates for the paper's single test fold (S4.1, "the fold with the
# most multi-point mutations"): f1 under a >=3 reading, f3 under >=2.  f0 is ruled out -- no
# assay survives the >=100-row filter in its >=3 slice, so Table 2's >=3 column could not exist.
#
# Writes oof_fold{F}_untrained.csv into a SEPARATE save_dir, so nothing here can touch the
# trained folds' outputs or the resume checkpoints f0/f3 will need.

set -euo pipefail

FOLD=$1

source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate h3ddg-reproduce
cd /ibex/user/guoj0f/H3-DDG/reproduce

SAVE_DIR=./results/bindinggym_untrained_fold${FOLD}

echo "=== untrained baseline | fold ${FOLD} | job ${SLURM_JOB_ID} ==="
python - <<'PY'
import torch, numpy, pandas, sklearn, Bio
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0))
print('numpy', numpy.__version__, '| pandas', pandas.__version__,
      '| sklearn', sklearn.__version__, '| biopython', Bio.__version__)
assert sklearn.__version__ in ('1.2.1', '1.3.2'), 'fold split is sklearn-version sensitive'
PY

for f in ./data/BindingGYM_cache/entries.pkl ./data/BindingGYM_cache/structures.pkl; do
  test -s "$f" || { echo "FATAL: missing cache $f"; exit 1; }
done

export WANDB_MODE=disabled
srun python train_bindinggym.py \
  --config_path ./config/train_h3-ddg_bindinggym.json \
  --test_fold "${FOLD}" \
  --tag untrained_baseline \
  --save_dir "${SAVE_DIR}" \
  --eval_only \
  --num_workers 12

echo "=== fold ${FOLD} untrained baseline DONE ==="
