#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/reproduce-finetune-proteinMPNN-over-BindingGYM/finetune_%x_%j.out
#SBATCH --error=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/reproduce-finetune-proteinMPNN-over-BindingGYM/finetune_%x_%j.err

# Fine-tune plain ProteinMPNN on BindingGYM's inter-assay cluster split -- ONE fold per job.
#
# This runs BindingGYM's own training/main.py, vendored into bgym_official/ with a single
# behavioural patch (--fold; see PATCHES.md). Everything that defines the protocol is theirs and
# untouched: same-assay batches, listMLE, AdamW 1e-3 / wd 0.05 / OneCycleLR, 256 steps per epoch,
# 100 epochs, patience 3, and epoch selection on the held-out fold's Spearman.
#
# All five folds share --tmp_path on purpose. Upstream writes one {DMS_id}_oof.csv per assay and
# each assay is held out in exactly one fold, so the five jobs assemble the complete
# out-of-fold set without a merge step. The log is per-fold (patched) so they do not truncate
# each other.

set -euo pipefail
FOLD=$1
TMP=${2:-inter_cluster_structure}

source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate bgym-official
cd /ibex/user/guoj0f/H3-DDG/reproduce/bgym_official/training

echo "=== fold ${FOLD} | tmp_path ${TMP} | job ${SLURM_JOB_ID} ==="
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

srun python main.py \
  --train_dms_mapping ../input/BindingGYM.csv \
  --dms_input ../input/Binding_substitutions_DMS \
  --structure_path ../input/structures \
  --model_type structure --mode inter --split cluster \
  --use_weight pretrained --batch_size 8 --seed 42 \
  --fold "${FOLD}" --tmp_path "${TMP}"

echo "=== fold ${FOLD} DONE ==="
