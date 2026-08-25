#!/bin/bash
# Build the env for this project ONLY. Follows bgym_official/install.sh, trimmed to what the
# structure path actually needs (no foldseek, no datasets, no EVE/SaProt/Tranception deps).
#
# ⚠ Never touch another project's env. Ibex already has `unibind` with PyG + torch_scatter, but
# ibex-usage §1d records a 2026-07-01 incident where one agent's env rebuild removed the conda env
# of a running experiment. A conda env belongs to one experiment.
#
# Version pins matter for two separate reasons:
#   torch 1.13.1+cu117 / torch-scatter 2.1.0+pt113cu117 / PyG 2.2.0 -- upstream's own combination,
#     and torch-scatter is a compiled extension that must match the torch+CUDA build exactly.
#   numpy 1.24.4 / scikit-learn 1.3.2 -- BindingGYM.yml's pins, and the pair that reproduces the
#     inter-assay fold assignment (verified: same md5 as the frozen tsv, diff 0 lines).
set -euo pipefail
ENV=bgym-official
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh

if conda env list | grep -qE "^${ENV}\s"; then
  echo "FATAL: env ${ENV} already exists. Refusing to rebuild -- inspect it instead."; exit 1
fi

conda create -y -n "$ENV" python=3.8
conda activate "$ENV"
python -V

pip install -q torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -q torch-scatter==2.1.0+pt113cu117 \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install -q torch-geometric==2.2.0
pip install -q fair-esm peft==0.12.0
# pinned LAST so nothing above can pull them forward
pip install -q "numpy==1.24.4" "scikit-learn==1.3.2" "pandas==2.0.3" "scipy==1.10.1" \
    "biopython==1.83" tqdm

echo "=== verification ==="
python - <<'PY'
import torch, torch_scatter, torch_geometric, numpy, sklearn, pandas, scipy, Bio, esm, peft
print('torch       ', torch.__version__, '| cuda', torch.version.cuda)
print('torch_scatter', torch_scatter.__version__)
print('torch_geometric', torch_geometric.__version__)
print('numpy       ', numpy.__version__)
print('sklearn     ', sklearn.__version__)
print('pandas      ', pandas.__version__)
print('scipy       ', scipy.__version__)
print('biopython   ', Bio.__version__)
print('peft        ', peft.__version__)
assert numpy.__version__ == '1.24.4' and sklearn.__version__ == '1.3.2', \
    'fold assignment depends on these -- see UPSTREAM.md'
PY
echo "ENV BUILD OK"
