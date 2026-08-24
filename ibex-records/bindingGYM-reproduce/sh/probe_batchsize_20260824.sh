#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:25:00
#SBATCH --job-name=bgym_probe
#SBATCH --output=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/probe_batchsize_%j.out
#SBATCH --error=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/probe_batchsize_%j.err

# Resolve the largest train/eval batch sizes that fit an a100, then exit.
#
# listMLE needs a slate of at least 2, and H3-DDG's A.4 caps the batch at "1, 2" -- so
# batch_size 2 is the single value that satisfies both, and there is nothing to fall back to if
# it does not fit. The local A4500 (20 GB, and sharing with another run) cannot answer this. A
# 25-minute job backfills almost immediately, whereas discovering it inside a 7h job that queued
# for days would waste the slot.

set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate h3ddg-reproduce
cd /ibex/user/guoj0f/H3-DDG/reproduce

python -c "import torch; p=torch.cuda.get_device_properties(0); print(f'GPU {p.name}  {p.total_memory/2**30:.1f} GiB')"

for FOLD in 0 2; do
  echo "########## probe fold ${FOLD}"
  python train_bindinggym_official.py \
    --config_path ./config/train_h3-ddg_bindinggym_strategy.json \
    --test_fold "${FOLD}" --save_dir "./results/probe_fold${FOLD}" \
    --probe_only --num_workers 8
done
echo "ALL PROBES DONE"
