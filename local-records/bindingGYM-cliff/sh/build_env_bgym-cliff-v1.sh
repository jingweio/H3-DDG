#!/usr/bin/env bash
# Pinned analysis env for BGYM-CLIFF v1.  CPU-only.  Do NOT install into h3ddg-reproduce.
#
# Why each pin is this value:
#   python 3.9.25   : matches h3ddg-reproduce, the env every verified number in this repo came from
#   numpy  1.22.4   : in make_inter_assay_folds.py's EXPECTED_NUMPY.  We only READ
#                     data_splits/inter_assay_folds.tsv, never recompute GroupKFold -- np.argsort's
#                     unstable quicksort silently changes the split across numpy versions -- but
#                     staying inside EXPECTED means any accidental recompute is still correct.
#   scipy  1.13.1   : cKDTree, sparse lsqr, optimize.least_squares, cluster.hierarchy.  Last
#                     series supporting py3.9 (1.14 requires >=3.10).
#   pandas 1.5.3    : usecols de-duplication of HLA-A2's repeated DMS_score column is verified here
#   sklearn 1.2.1   : IsotonicRegression, Ridge, KFold.  In EXPECTED_SKLEARN.
#   biopython 1.81  : Bio.PDB.SASA.ShrakeRupley -- no DSSP, no freesasa needed
#   matplotlib 3.7.5: vector PDF + 600 dpi PNG.  Absent from h3ddg-reproduce, hence a new env.
# statsmodels is deliberately NOT a dependency: BH-FDR, HC3 SEs, the Poisson/binomial GLM by IRLS
# and the mixture EM are ~20 lines each on top of scipy.
set -euo pipefail
source /home/guoj0f/anaconda3/etc/profile.d/conda.sh
conda env list | awk '{print $1}' | grep -qx bgym-cliff-v1 || conda create -y -n bgym-cliff-v1 python=3.9.25
conda activate bgym-cliff-v1
pip install --no-cache-dir \
  numpy==1.22.4 scipy==1.13.1 pandas==1.5.3 scikit-learn==1.2.1 \
  biopython==1.81 matplotlib==3.7.5 pytest==7.4.4
python - <<'PY'
import sys, numpy, scipy, pandas, sklearn, Bio, matplotlib
got = (sys.version.split()[0], numpy.__version__, scipy.__version__,
       pandas.__version__, sklearn.__version__, Bio.__version__, matplotlib.__version__)
want = ('3.9.25','1.22.4','1.13.1','1.5.3','1.2.1','1.81','3.7.5')
assert got == want, f'env pin mismatch: {got} != {want}'
import scipy.linalg, scipy.spatial, scipy.sparse.linalg, scipy.optimize, scipy.cluster.hierarchy
from sklearn.isotonic import IsotonicRegression
from Bio.PDB.SASA import ShrakeRupley
print('[env] bgym-cliff-v1 OK', got)
PY
pip list --format=freeze > "$(dirname "$0")/env_bgym-cliff-v1_freeze.txt"
