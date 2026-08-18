#!/bin/bash
#SBATCH --job-name=skempiv2_cv3_h3ddg
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/skempiv2_cv3_h3ddg_20260817-092000_%j.out
#SBATCH --error=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/skempiv2_cv3_h3ddg_20260817-092000_%j.err

set -euo pipefail

source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate h3ddg-reproduce

cd /ibex/user/guoj0f/H3-DDG/reproduce

echo "=== env ==="
python - <<'PY'
import torch, numpy, pandas, sklearn, Bio
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0))
print('numpy', numpy.__version__, '| pandas', pandas.__version__,
      '| sklearn', sklearn.__version__, '| biopython', Bio.__version__)
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "=== config ==="
cat ./config/train_h3-ddg.json

# 前置验证：SKEMPI v2 3-fold CV,严格用作者 repo 的 config(lr 4e-5 / max_iter 38000 / 3 folds)
export WANDB_MODE=disabled
srun python train_skempi.py \
  --config_path ./config/train_h3-ddg.json \
  --tag cv3_h3ddg_repro

echo "=== DONE train; running validate_all over the saved per-fold results ==="
SAVE_DIR=$(ls -td ./results/*_cv3_h3ddg_repro | sed -n '1p')
echo "save_dir=${SAVE_DIR}"
python validate_all.py --save_dir "${SAVE_DIR}" --top_k 5
echo "=== ALL DONE ==="
