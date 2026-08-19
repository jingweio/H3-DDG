#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --output=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/bindinggym_perfold_20260819_f%x_%j.out
#SBATCH --error=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/bindinggym_perfold_20260819_f%x_%j.err

# One INDEPENDENT job per fold of the BindingGYM inter-assay protocol.
# Fold and walltime come from the submitter (sh/submit_bindinggym_perfold.sh), which sets
# --job-name and --time per fold; walltime is sized to each fold's own measured cost instead of
# the single worst case, so the cheap folds queue like cheap jobs.
#
# Replaces array 50674363_[0-4] (5 x 23h), whose uniform 23h walltime and 5-task shape made it
# hard for the backfill scheduler to place -- it sat at an estimated 3-4 days out.
#
# Safe against a walltime kill: train_bindinggym.py now writes a resume point every ckpt_freq
# iterations, so requeueing with --resume continues instead of restarting.

set -euo pipefail

FOLD=$1

source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate h3ddg-reproduce
cd /ibex/user/guoj0f/H3-DDG/reproduce

SAVE_DIR=./results/bindinggym_interassay_fold${FOLD}

echo "=== fold ${FOLD} | env ==="
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
  --tag h3ddg_interassay \
  --save_dir "${SAVE_DIR}" \
  --resume \
  --num_workers 12

echo "=== fold ${FOLD} DONE ==="
