#!/bin/bash
#SBATCH --job-name=skempi_f
#SBATCH --array=0-2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/skempiv2_cv3_h3ddg_20260817-092000_f%a_%A.out
#SBATCH --error=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/skempiv2_cv3_h3ddg_20260817-092000_f%a_%A.err

# SKEMPI v2 3-fold CV, ONE JOB PER FOLD.
#
# WHY SPLIT: as a single 3-fold job this needs a 48h walltime, which excludes it from Ibex's
# gpu24 partition (24h limit, 32 a100 nodes -- the largest a100 pool; gpu has 17, gpu72 has 18).
# SLURM's estimated start for the 48h job was 3 days out. At 14h each, all three partitions are
# eligible and the folds run in parallel: ~10h wall-clock instead of 3 days + 30h.
#
# The folds are fully independent -- CrossValidation holds a separate model/optimizer/scheduler
# per fold and train() never crosses them.
# KNOWN DEVIATION (user-accepted): the released code calls set_seed(42) once before the 3-fold
# loop, so fold 1 and 2 continued the RNG stream left by the previous fold. Split into separate
# processes, every fold starts from a fresh seed-42 state, so each fold sees a different training
# sample order than the monolithic run would have given it. Any seed is equally valid, but this
# is a real difference from the released code and is recorded as such.
#
# All three tasks SHARE ${SAVE_DIR} so that validate_all.py can later combine their
# fold{F}_its{ITS}_results.csv files. train_skempi.py --save_dir deliberately avoids
# utils.check_dir() there, because check_dir(overwrite=True) does shutil.rmtree and would wipe a
# sibling fold's finished checkpoints.

set -euo pipefail

source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate h3ddg-reproduce

cd /ibex/user/guoj0f/H3-DDG/reproduce

FOLD=${SLURM_ARRAY_TASK_ID}
SAVE_DIR=./results/skempiv2_cv3_h3ddg_20260817-092000

echo "=== fold ${FOLD} | env ==="
python - <<'PY'
import torch, numpy, pandas, sklearn, Bio
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0))
print('numpy', numpy.__version__, '| pandas', pandas.__version__,
      '| sklearn', sklearn.__version__, '| biopython', Bio.__version__)
PY
echo "=== config ==="
cat ./config/train_h3-ddg.json

export WANDB_MODE=disabled
srun python train_skempi.py \
  --config_path ./config/train_h3-ddg.json \
  --tag cv3_h3ddg_repro \
  --fold "${FOLD}" \
  --save_dir "${SAVE_DIR}"

echo "=== fold ${FOLD} DONE ==="
