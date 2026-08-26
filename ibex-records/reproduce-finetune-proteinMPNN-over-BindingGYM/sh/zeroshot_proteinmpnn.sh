#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=pmpnn_zs
#SBATCH --output=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/reproduce-finetune-proteinMPNN-over-BindingGYM/zeroshot_%j.out
#SBATCH --error=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/reproduce-finetune-proteinMPNN-over-BindingGYM/zeroshot_%j.err

# Deliverable (1): zero-shot pretrained ProteinMPNN over all 25 assays.
#
# Independent of the fine-tuning arm in every way that matters: no training, no folds (zero-shot
# scores the whole dataset once), and it uses the VANILLA ProteinMPNN under baselines/ with the
# positional forward(X, S, ...) -- so it needs neither torch_geometric nor esm/peft, unlike
# training/main.py which imports all of them at module level.
#
# Idempotent by design: one output csv per assay, and an assay whose csv already exists is
# skipped. A timeout is therefore resumed by simply resubmitting -- which matters because the
# upstream script has no checkpointing of its own and --batch_size defaults to 1, so throughput
# over 376,446 rows is the open unknown here.

set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate bgym-official
cd /ibex/user/guoj0f/H3-DDG/reproduce/bgym_official/baselines/protein_mpnn

OUT=/ibex/user/guoj0f/H3-DDG/reproduce/bgym_official/output/zeroshot
IN=/ibex/user/guoj0f/H3-DDG/reproduce/bgym_official/input
mkdir -p "$OUT"

python -c "import torch;p=torch.cuda.get_device_properties(0);print(f'GPU {p.name} {p.total_memory/2**30:.0f} GiB')"

# assay order is BindingGYM.csv's row order, which is what --dms_index indexes
mapfile -t IDS < <(python -c "
import pandas as pd
print('\n'.join(pd.read_csv('$IN/BindingGYM.csv')['DMS_id']))")
echo "assays: ${#IDS[@]}"

for i in $(seq 0 $((${#IDS[@]}-1))); do
  ID="${IDS[$i]}"
  if [ -s "$OUT/${ID}.csv" ]; then
    echo "[$i] $ID  -- already scored, skipping"
    continue
  fi
  echo "=== [$i] $ID  $(date +%H:%M:%S) ==="
  srun python compute_fitness_multi_pdb.py \
    --dms_mapping "$IN/BindingGYM.csv" \
    --dms_input "$IN/Binding_substitutions_DMS" \
    --structure_folder "$IN/structures" \
    --dms_index "$i" \
    --model_location ../../training/cache/v_48_020.pt \
    --dms_output "$OUT" \
    --backbone_noise 0.00 \
    --suppress_print 1
done

echo "=== scored: $(ls "$OUT"/*.csv 2>/dev/null | wc -l) / ${#IDS[@]} ==="
echo ZEROSHOT_DONE
