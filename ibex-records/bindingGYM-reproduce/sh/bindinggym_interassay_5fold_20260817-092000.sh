#!/bin/bash
#SBATCH --job-name=bgym_interassay
#SBATCH --array=0-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/bindinggym_interassay_5fold_20260817-092000_f%a_%A.out
#SBATCH --error=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/bindinggym_interassay_5fold_20260817-092000_f%a_%A.err

# One array task per fold of the official BindingGYM inter-assay 5-fold protocol.
# Task k trains on the 4 cluster-groups other than k and evaluates the whole of group k,
# so the 5 tasks together produce out-of-fold predictions for all 25 assays / 376,446 rows.
#
# PREREQUISITE: ./data/BindingGYM_cache/{entries,structures}.pkl must already exist.
# The cache is built once (locally, then rsync'd) precisely so the 5 array tasks do not race
# each other writing the same pickle.

set -euo pipefail

source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate h3ddg-reproduce

cd /ibex/user/guoj0f/H3-DDG/reproduce

FOLD=${SLURM_ARRAY_TASK_ID}

echo "=== fold ${FOLD} | env ==="
python - <<'PY'
import torch, numpy, pandas, sklearn, Bio
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0))
print('numpy', numpy.__version__, '| pandas', pandas.__version__,
      '| sklearn', sklearn.__version__, '| biopython', Bio.__version__)
assert sklearn.__version__ in ('1.2.1', '1.3.2'), 'fold split is sklearn-version sensitive'
PY

for f in ./data/BindingGYM_cache/entries.pkl ./data/BindingGYM_cache/structures.pkl; do
  test -s "$f" || { echo "FATAL: missing cache $f -- build it before submitting the array"; exit 1; }
done
echo "=== frozen fold assignment ==="
cat ./data_splits/inter_assay_folds.tsv

export WANDB_MODE=disabled
srun python train_bindinggym.py \
  --config_path ./config/train_h3-ddg_bindinggym.json \
  --test_fold "${FOLD}" \
  --tag h3ddg_interassay \
  --num_workers 12

echo "=== fold ${FOLD} DONE ==="
