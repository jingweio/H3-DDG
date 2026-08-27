#!/bin/bash
# H3-DDG × BindingGYM, readout-correction arm (logitdiff), fold 1 -- on the workstation A100.
#
# Why this arm exists: H3-DDG's thermodynamic cycle presumes the label is a binding free-energy
# change, and BindingGYM's labels are DMS fitness / enrichment scores that carry no such meaning.
# This run swaps in BindingGYM's own readout (log p(mut|backbone) - log p(wt|backbone), one
# forward on the complex) and keeps exactly one thing from H3-DDG besides the architecture:
# backbone_noise 0.0.
#
# Moved off Ibex because it never got a slot there. No scheduler here, so it runs to completion
# in one go -- ~7.8h expected, against Ibex's 2:30 slots that would have needed four rounds.
set -euo pipefail

source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate h3ddg-reproduce

ROOT=/home/guoj0f/repos/H3-DDG/reproduce
cd "$ROOT"
echo "[synced_commit] $(head -1 .synced_commit 2>/dev/null)"
echo "[dirty]         $(tail -n +2 .synced_commit 2>/dev/null | wc -l) uncommitted file(s) at sync time"

# BindingGYM's raw benchmark data comes from the shared store, not from this branch (skill §5):
# it is fixed, public, identical across branches and 1.2 GB. bindinggym_dataset.py reads this
# env var. The in-branch ./data/input fallback no longer exists -- it was a byte-identical
# duplicate (73/73 md5) and was deleted, so this export is load-bearing, not a convenience.
export BINDINGGYM_INPUT=/data/guoj0f/share/BindingGYM/input
test -d "$BINDINGGYM_INPUT/Binding_substitutions_DMS" || { echo "FATAL: shared BindingGYM data missing"; exit 1; }
echo "[data] $BINDINGGYM_INPUT  ("$(ls $BINDINGGYM_INPUT/Binding_substitutions_DMS | wc -l)" DMS csv, "$(ls $BINDINGGYM_INPUT/structures | wc -l)" structures)"

# Single A100 -- do NOT set CUDA_VISIBLE_DEVICES (skill §1).
python - <<'PY'
import torch, numpy, sklearn
n = torch.cuda.get_device_name(0); p = torch.cuda.get_device_properties(0)
print(f'GPU: {n}  {p.total_memory/2**30:.0f} GiB | torch {torch.__version__} cuda {torch.version.cuda}')
print(f'numpy {numpy.__version__} | sklearn {sklearn.__version__}')
assert 'A100' in n, n
assert (numpy.__version__, sklearn.__version__) == ('1.22.4', '1.2.1'), \
    'the inter-assay fold assignment depends on exactly these -- see data_splits/'
PY

# Shared machine: record who else is on the card at launch, so an OOM later is attributable.
echo "--- GPU occupancy at launch ---"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
df -h /home /data | tail -2

# Outputs stay in the worktree: ~28 MB x 5 checkpoints + one resume point + 5 OOF csvs, a few
# hundred MB total. Well inside the 5 GB /home threshold (skill 2), so no split-tree needed.
SAVE_DIR="$ROOT/results/bindinggym_logitdiff_fold1"
mkdir -p "$SAVE_DIR"

python train_bindinggym_official.py \
  --config_path ./config/train_h3-ddg_bindinggym_logitdiff.json \
  --test_fold 1 \
  --save_dir "$SAVE_DIR" \
  --resume \
  --num_workers 12

echo "EXIT=$?"
echo "=== logitdiff fold1 DONE ==="
