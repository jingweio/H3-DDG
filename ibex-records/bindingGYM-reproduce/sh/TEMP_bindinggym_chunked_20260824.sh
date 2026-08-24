#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --output=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/TEMP_chunked_20260824_%x_%j.out
#SBATCH --error=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/TEMP_chunked_20260824_%x_%j.err

# ============================================================================
#  TEMPORARY -- QUEUE WORKAROUND ONLY.  NOT part of the reproduction pipeline.
# ============================================================================
# The normal per-fold script is sh/bindinggym_perfold_20260819.sh; f1/f2/f4 were produced with
# it and it stays the canonical one.  This file exists ONLY because f0/f3 asked for 18h/16h and
# then sat in the queue for 5+ days: the three completed folds each finished in <= 4:05:46, so
# those walltimes were ~4x over-requested and the backfill scheduler could not place them.
#
# What this does differently: a SHORT walltime (6-7h) that may not cover the whole run, relying
# on --resume to pick the run up in a second chunk.  Nothing about the training itself changes --
# same config, same fold split, same code path -- only the walltime and the "may take 2 jobs"
# expectation.  DELETE THIS FILE once f0/f3 are done.
#
# Correctness of chunking: train_bindinggym.py writes a resume point every ckpt_freq (5000) iters
# and, on finishing, one at max_iter-1.  A resumed job therefore either continues training or --
# if training already completed -- skips straight to the full held-out evaluation.  The eval
# itself is NOT checkpointed, so a kill during eval costs a re-run of the eval, not of training.

set -euo pipefail

FOLD=$1

source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate h3ddg-reproduce
cd /ibex/user/guoj0f/H3-DDG/reproduce

SAVE_DIR=./results/bindinggym_interassay_fold${FOLD}
CKPT=${SAVE_DIR}/checkpoint/resume_fold${FOLD}.pt

echo "=== fold ${FOLD} | chunked run | job ${SLURM_JOB_ID} | walltime ${SLURM_JOB_END_TIME:-?} ==="

# Self-document where this chunk picks up, so the log says which chunk it is without cross-
# referencing another job's output.
if [ -f "${CKPT}" ]; then
  python - "${CKPT}" <<'PY'
import sys, torch
b = torch.load(sys.argv[1], map_location='cpu')
print(f"[chunk] resuming: checkpoint at iteration {b['iteration']} / max_iter {b['max_iter']}")
PY
else
  echo "[chunk] no checkpoint -- this is chunk 1, starting from iteration 0"
fi

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

# Only reached if srun exited cleanly. A walltime kill leaves this unprinted -- that absence,
# plus the last "[ckpt] saved resume point at iteration N" line, is the signal to submit chunk 2.
echo "=== fold ${FOLD} FULLY DONE (no second chunk needed) ==="
