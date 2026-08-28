#!/bin/bash
# BindingGYM official ProteinMPNN finetune, inter-assay cluster split -- folds 3 then 4, SERIAL,
# on the workstation A100.
#
# Why not Ibex: the queue stopped rewarding short walltimes (measured: 2:30 / 8h / 16h / 24h all
# return the SAME estimated start, ~9 days out), so the chunked --resume strategy that produced
# f1/f2 now costs a full queue cycle per 3:30 of compute. Here there is no scheduler and no
# walltime, so each fold runs straight through to its own patience-3 early stop in one go and
# --resume is unnecessary.
#
# Upstream selection rule is UNCHANGED for these two folds: patience-3 on the held-out fold,
# identical to how f1/f2 were produced. Only fold 0 uses a temporary threshold-stop variant.
set -euo pipefail

source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate bgym-official

ROOT=/home/guoj0f/repos/H3-DDG/reproduce
cd "$ROOT/bgym_official/training"
echo "[synced_commit] $(head -1 "$ROOT/.synced_commit" 2>/dev/null)"

# Benchmark data from the shared store. main.py takes all three as CLI args, so no second copy of
# the 595 MB input/ tree is shipped; only ./cache/{BindingGYM_cluster.tsv,v_48_020.pt} have to sit
# relative to this working directory (main.py:149 and :407 hardcode those two).
IN=/data/guoj0f/share/BindingGYM/input
test -d "$IN/Binding_substitutions_DMS" || { echo "FATAL: shared BindingGYM data missing"; exit 1; }
test -s ./cache/v_48_020.pt            || { echo "FATAL: missing ProteinMPNN weights"; exit 1; }
test -s ./cache/BindingGYM_cluster.tsv || { echo "FATAL: missing cluster table"; exit 1; }

# These two folds start FROM SCRATCH. Nothing from the Ibex runs was copied over (the rsync
# excluded training/output and output), so assert that rather than trust it -- resuming fold 3
# from fold 1's state would fail silently and produce a plausible wrong number.
for f in 3 4; do
  find . -name "resume_fold${f}.pt" -o -name "model${f}.ckpt" | grep -q . \
    && { echo "FATAL: pre-existing state for fold ${f}"; exit 1; }
done
echo "[clean] no pre-existing state for folds 3/4"

python -c "
import torch, numpy, sklearn
n = torch.cuda.get_device_name(0); print('GPU:', n); assert 'A100' in n, n
print('numpy', numpy.__version__, '| sklearn', sklearn.__version__)
assert (numpy.__version__, sklearn.__version__) == ('1.24.4','1.3.2'), \
    'the inter-assay fold assignment depends on these -- see bgym_official/UPSTREAM.md'
"
echo "--- GPU occupancy at launch (shared machine) ---"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
df -h /home | tail -1

for FOLD in 3 4; do
  echo ""
  echo "=================== fold ${FOLD} start $(date '+%F %T') ==================="
  python main.py \
    --train_dms_mapping "$IN/BindingGYM.csv" \
    --dms_input "$IN/Binding_substitutions_DMS" \
    --structure_path "$IN/structures" \
    --model_type structure --mode inter --split cluster \
    --use_weight pretrained --batch_size 8 --seed 42 \
    --fold "${FOLD}" --tmp_path inter_cluster_structure
  echo "=================== fold ${FOLD} done  $(date '+%F %T') ==================="
done

echo "EXIT=$?"
echo "=== pmpnn f3+f4 ALL DONE ==="
