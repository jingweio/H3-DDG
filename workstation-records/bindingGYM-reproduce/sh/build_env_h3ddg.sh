#!/bin/bash
# Build the h3ddg-reproduce env on the workstation. Mirrors the local env of the same name
# (workstation-usage §10: env name mirrors the local project env).
#
# 🔴 numpy 1.22.4 + scikit-learn 1.2.1 are NOT ordinary pins. The BindingGYM inter-assay fold
# assignment comes from GroupKFold, whose tie order is decided by numpy's unstable argsort, and
# 12 of the 14 cluster weights are tied. A different numpy silently produces a DIFFERENT split
# with no error -- the incident recorded in the skill's §3-0. This pair is the one verified to
# reproduce the frozen split (md5 d23e15f9...), and make_inter_assay_folds.py asserts it.
#
# torch 1.13.1+cu117: the driver here caps at CUDA 12.2, and cu117 wheels run fine on it.
# Never use the system nvcc (CUDA 10.1) -- pip wheels carry their own runtime (§10).
set -euo pipefail
ENV=h3ddg-reproduce
source /data/guoj0f/miniconda3/etc/profile.d/conda.sh

if conda env list | grep -qE "^${ENV}\s"; then
  if [ "${1:-}" = "--force" ]; then
    echo "removing existing ${ENV} (THIS project's env only)"; conda env remove -y -n "$ENV"
  else
    echo "FATAL: ${ENV} already exists. Pass --force to rebuild, or inspect it."; exit 1
  fi
fi

conda create -y -n "$ENV" python=3.9
conda activate "$ENV"
python -V

pip install -q torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
# pinned AFTER torch so nothing above can pull them forward
pip install -q "numpy==1.22.4" "pandas==1.5.3" "scikit-learn==1.2.1" "scipy==1.13.1" \
    "biopython==1.81" tqdm

echo "=== verification ==="
python - <<'PY'
import sys, torch, numpy, pandas, sklearn, scipy, Bio
print('python  ', sys.version.split()[0])
print('torch   ', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu     ', torch.cuda.get_device_name(0))
print('numpy   ', numpy.__version__)
print('pandas  ', pandas.__version__)
print('sklearn ', sklearn.__version__)
print('scipy   ', scipy.__version__)
print('biopython', Bio.__version__)
assert (numpy.__version__, sklearn.__version__) == ('1.22.4', '1.2.1'), \
    'the inter-assay fold assignment depends on exactly these -- see the header'
PY
echo "ENV BUILD OK"
