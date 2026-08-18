#!/bin/bash
#SBATCH --job-name=skempi_valall
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/skempiv2_validate_all_20260817-092000_%j.out
#SBATCH --error=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/bindingGYM-reproduce/skempiv2_validate_all_20260817-092000_%j.err

# Step 2 of the SKEMPI reproduction: combine the 3 per-fold jobs' saved per-iteration results.
# Submit with --dependency=afterok:<array job id> so it only runs once all three folds finished.
# CPU-only: validate_all.py just re-reads fold{F}_its{ITS}_results.csv and recomputes metrics.
#
# validate_all.py picks the top-k iterations per fold BY THAT FOLD'S OWN VALIDATION SPEARMAN and
# then enumerates all k^3 combinations. Since SKEMPI's val fold IS its test fold, this is model
# selection on the evaluation data -- the authors' protocol, reproduced as-is and flagged in the
# record md.

set -euo pipefail

source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate h3ddg-reproduce

cd /ibex/user/guoj0f/H3-DDG/reproduce
SAVE_DIR=./results/skempiv2_cv3_h3ddg_20260817-092000

echo "=== per-fold artifacts present ==="
# NOTE: `... | head -N` is a trap under `set -euo pipefail`: head closes the pipe after N lines,
# ls dies of SIGPIPE, pipefail propagates it, and set -e kills the script before it does any work.
# That is exactly how job 50673544 failed in 4 seconds. sed reads its input to the end instead.
ls -la "${SAVE_DIR}" | sed -n '1,20p'
for f in 0 1 2; do
  n=$(ls "${SAVE_DIR}"/fold${f}_its*_results.csv 2>/dev/null | wc -l)
  echo "fold${f}: ${n} saved per-iteration result files"
  test "${n}" -gt 0 || { echo "FATAL: fold${f} produced no per-iteration results"; exit 1; }
done

python validate_all.py --save_dir "${SAVE_DIR}" --top_k 5
echo "=== best combination (max all-mode Pearson) ==="
python - <<'PY'
import re, os
p = './results/skempiv2_cv3_h3ddg_20260817-092000/validate_all.txt'
blocks, cur = [], []
for line in open(p):
    if '| [val]' in line and cur:
        blocks.append(cur); cur = [line]
    else:
        cur.append(line)
if cur:
    blocks.append(cur)
best = None
for b in blocks:
    t = ''.join(b)
    m = re.search(r'Mode all: A-Pea ([\d.]+) A-Spe ([\d.]+) \| RMSE ([\d.]+) MAE ([\d.]+) '
                  r'AUROC ([\d.]+) AUPRC ([\d.]+) \| P-Pea ([\d.]+) P-Spe ([\d.]+)', t)
    if m and (best is None or float(m.group(1)) > best[0]):
        best = (float(m.group(1)), t.strip())
if best:
    print(best[1])
    print('\n--- paper Table 1 (H3-DDG, all) ---')
    print('A-Pea 0.7501  A-Spe 0.6604  RMSE 1.3665  AUROC 0.7920  P-Pea 0.5686  P-Spe 0.5281')
else:
    print('no parsable combination found in', p)
PY
echo "=== ALL DONE ==="
